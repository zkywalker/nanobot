"""Channel manager for coordinating chat channels."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Config


class ChannelManager:
    """
    Manages chat channels and coordinates message routing.

    Responsibilities:
    - Initialize enabled channels (Telegram, WhatsApp, etc.)
    - Start/stop channels
    - Route outbound messages
    """

    def __init__(self, config: Config, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task | None = None

        self._init_channels()

    def _init_channels(self) -> None:
        """Initialize channels discovered via pkgutil scan.

        Built-in channels have typed config fields on ``ChannelsConfig``.
        Plugin channels (symlinked in) may declare a ``config_class`` attribute
        on their channel class; the manager will construct the config from the
        raw dict that ``ChannelsConfig(extra="allow")`` preserves from JSON.
        """
        from pydantic.alias_generators import to_camel

        from nanobot.channels.registry import discover_channel_names, load_channel_class

        groq_key = self.config.providers.groq.api_key

        for modname in discover_channel_names():
            # Built-in channels: snake_case attr.  Plugin extras: camelCase key.
            section = getattr(self.config.channels, modname, None)
            if section is None:
                section = getattr(self.config.channels, to_camel(modname), None)

            # Plugin channel: section is a raw dict from extra="allow".
            # Load the class first to get its config_class, then parse.
            if isinstance(section, dict):
                if not section.get("enabled", False):
                    continue
                try:
                    cls = load_channel_class(modname)
                except ImportError as e:
                    logger.warning("{} channel not available: {}", modname, e)
                    continue
                config_cls = getattr(cls, "config_class", None)
                if config_cls is None:
                    logger.warning(
                        "{} channel has no config_class, skipping", modname
                    )
                    continue
                try:
                    section = config_cls(**section)
                except Exception as e:
                    logger.error("{} channel config error: {}", modname, e)
                    continue
            else:
                # Built-in channel with typed config field
                if not section or not getattr(section, "enabled", False):
                    continue
                try:
                    cls = load_channel_class(modname)
                except ImportError as e:
                    logger.warning("{} channel not available: {}", modname, e)
                    continue

            try:
                channel = cls(section, self.bus)
                channel.transcription_api_key = groq_key
                self.channels[modname] = channel
                logger.info("{} channel enabled", cls.display_name)
            except Exception as e:
                logger.warning("{} channel failed to init: {}", modname, e)

        self._validate_allow_from()

    def _validate_allow_from(self) -> None:
        for name, ch in self.channels.items():
            if getattr(ch.config, "allow_from", None) == []:
                raise SystemExit(
                    f'Error: "{name}" has empty allowFrom (denies all). '
                    f'Set ["*"] to allow everyone, or add specific user IDs.'
                )

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        """Start a channel and log any exceptions."""
        try:
            await channel.start()
        except Exception as e:
            logger.error("Failed to start channel {}: {}", name, e)

    async def start_all(self) -> None:
        """Start all channels and the outbound dispatcher."""
        if not self.channels:
            logger.warning("No channels enabled")
            return

        # Start outbound dispatcher
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        # Start channels
        tasks = []
        for name, channel in self.channels.items():
            logger.info("Starting {} channel...", name)
            tasks.append(asyncio.create_task(self._start_channel(name, channel)))

        # Wait for all to complete (they should run forever)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """Stop all channels and the dispatcher."""
        logger.info("Stopping all channels...")

        # Stop dispatcher
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        # Stop all channels
        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info("Stopped {} channel", name)
            except Exception as e:
                logger.error("Error stopping {}: {}", name, e)

    async def _dispatch_outbound(self) -> None:
        """Dispatch outbound messages to the appropriate channel."""
        logger.info("Outbound dispatcher started")

        while True:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_outbound(),
                    timeout=1.0
                )

                if msg.metadata.get("_progress"):
                    if msg.metadata.get("_tool_hint") and not self.config.channels.send_tool_hints:
                        continue
                    if not msg.metadata.get("_tool_hint") and not self.config.channels.send_progress:
                        continue

                channel = self.channels.get(msg.channel)
                if channel:
                    try:
                        await channel.send(msg)
                    except Exception as e:
                        logger.error("Error sending to {}: {}", msg.channel, e)
                else:
                    logger.warning("Unknown channel: {}", msg.channel)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def get_channel(self, name: str) -> BaseChannel | None:
        """Get a channel by name."""
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """Get status of all channels."""
        return {
            name: {
                "enabled": True,
                "running": channel.is_running
            }
            for name, channel in self.channels.items()
        }

    def get_channel_tools(self) -> list:
        """Collect tools from all enabled channels."""
        tools = []
        for ch in self.channels.values():
            tools.extend(ch.get_tools())
        return tools

    @property
    def enabled_channels(self) -> list[str]:
        """Get list of enabled channel names."""
        return list(self.channels.keys())
