# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
Compatibility layer for decord using PyAV.
This module provides a drop-in replacement for decord's VideoReader.
"""

import numpy as np

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

try:
    from decord import VideoReader as DecordVideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

    def cpu(device_id=0):
        """Dummy cpu context for compatibility."""
        return None


class VideoReader:
    """
    A drop-in replacement for decord.VideoReader using PyAV.
    """

    def __init__(self, uri, ctx=None, width=None, height=None):
        if HAS_DECORD:
            # Use decord if available
            if width is not None and height is not None:
                self._reader = DecordVideoReader(uri, ctx=ctx, width=width, height=height)
            elif ctx is not None:
                self._reader = DecordVideoReader(uri, ctx=ctx)
            else:
                self._reader = DecordVideoReader(uri)
            self._use_decord = True
        elif HAS_PYAV:
            self._use_decord = False
            self._uri = uri
            self._width = width
            self._height = height
            self._frames = None
            self._container = None
            self._load_video()
        else:
            raise ImportError("Neither decord nor PyAV is available. Please install one of them.")

    def _load_video(self):
        """Load all frames from the video using PyAV."""
        self._container = av.open(self._uri)
        self._frames = []

        stream = self._container.streams.video[0]

        for frame in self._container.decode(video=0):
            img = frame.to_ndarray(format='rgb24')

            # Resize if needed
            if self._width is not None and self._height is not None:
                import cv2
                img = cv2.resize(img, (self._width, self._height))

            self._frames.append(img)

        self._container.close()

    def __len__(self):
        if self._use_decord:
            return len(self._reader)
        return len(self._frames)

    def __getitem__(self, idx):
        if self._use_decord:
            return self._reader[idx]

        frame = self._frames[idx]
        return FrameWrapper(frame)

    def __iter__(self):
        if self._use_decord:
            return iter(self._reader)
        return iter(FrameWrapper(f) for f in self._frames)

    def next(self):
        """Get the first frame (used to get video dimensions)."""
        if self._use_decord:
            return self._reader.next()
        return FrameWrapper(self._frames[0])


class FrameWrapper:
    """Wrapper to make numpy arrays behave like decord frames."""

    def __init__(self, array):
        self._array = array

    def asnumpy(self):
        return self._array

    @property
    def shape(self):
        return self._array.shape

    def permute(self, *dims):
        """Permute dimensions (for torch compatibility)."""
        import torch
        return torch.from_numpy(self._array).permute(*dims)
