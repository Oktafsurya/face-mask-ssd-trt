import asyncio
import json
import mimetypes
from aiohttp import web
from aiohttp.web_runner import GracefulExit
from aiohttp_middlewares import cors_middleware
from typing import Optional

FACEMASK_HOST = '0.0.0.0'
FACEMASK_PORT = 9473

SUPPORTED_MEDIA = ["video"]
RTSP_SCHEME = 'rtsp://'
USB_CAM_SCHEME = '/dev/video'

class MediaType:
    def __init__(self, type: Optional[str]):
        self._type = type

    def __eq__(self, media_type):
        return self.type == media_type.type

    @property
    def type(self):
        return self._type

def register_mediatype(media_type):
    setattr(media_type, "NONE", None)
    for media in SUPPORTED_MEDIA:
        type = MediaType(media)
        setattr(media_type, media.upper(), type)
    return media_type

def _media_inspector(media_name: str) -> str:
    mimestart = mimetypes.guess_type(media_name)[0]
    if mimestart != None:
        mimestart = mimestart.split('/')[0]
        return mimestart if mimestart in SUPPORTED_MEDIA else None
    elif media_name.startswith((RTSP_SCHEME, USB_CAM_SCHEME)):
        return "video"
    return None

@register_mediatype
class MEDIA_TYPE:
    pass

class StreamHeader:
    def __getitem__(self, boundary: str):
        return {'Content-Type': 'multipart/x-mixed-replace; boundary={}'.format(boundary)}

STREAM_HEADER = StreamHeader()

class StreamHandler:
    def __init__(self, boundary: str):
        self._boundary = boundary
        self._response = web.StreamResponse(status=200,
                                            reason='OK',
                                            headers=STREAM_HEADER[boundary])
    @property
    def response(self):
        return self._response

    async def prepare(self, request):
        await self._response.prepare(request)

    async def write(self, data, content_type: str):
        await self._response.write('--{}\r\n'.format(self._boundary).encode('utf-8'))
        await self._response.write('Content-Type: {}\r\n'.format(content_type).encode('utf-8'))
        await self._response.write('Content-Length: {}\r\n'.format(len(data)).encode('utf-8'))
        await self._response.write(b"\r\n")
        await self._response.write(data)
        await self._response.write(b"\r\n")

class FaceMaskServer:
    def __init__(self, _io_handler):
        self._io_handler = _io_handler
        self.initialize()

    async def index(self, request):
        return web.Response(content_type='application/json',
                            text="Face mask detection using SSD and running on TensorRT")

    async def stop(self, request):
        await self._app.shutdown()
        await self._app.cleanup()
        del self._app
        raise GracefulExit

    async def analytics(self, request):
        try:
            boundary = "data"
            content_type = "application/json"

            handler = StreamHandler(boundary)
            await handler.prepare(request)

            while True:
                data = self._io_handler.info
                data = json.dumps(data).encode('utf-8')              
                await asyncio.sleep(0.02)
                await handler.write(data, content_type)

            return handler.response

        except asyncio.CancelledError:
            # raise Exception('An error occurred when trying to process analytics data')
            pass

    async def output(self, request):
        media_type = getattr(MEDIA_TYPE, str(_media_inspector(self._io_handler.source)).upper())
        if media_type == MEDIA_TYPE.VIDEO:
            try:
                boundary = "frame"
                content_type = "image/jpeg"

                handler = StreamHandler(boundary)
                await handler.prepare(request)

                while True:
                    try:
                        data = self._io_handler.inference()
                        await asyncio.sleep(0.02)
                        if request.protocol.transport.is_closing():
                            continue
                        await handler.write(data, content_type)
                    except ConnectionResetError:
                        # handler = StreamHandler(boundary)
                        # await handler.prepare(request)
                        pass
        
                return handler.response

            except asyncio.CancelledError:
                pass

    def initialize(self):
        self._app = web.Application(middlewares=[cors_middleware(allow_all=True)])
        self._app.router.add_get('/', self.index)
        self._app.router.add_get('/frame-result', self.output)
        self._app.router.add_get('/analytics', self.analytics)
        self._app.router.add_get('/stop', self.stop)

    def run(self):
        """Run the elangai server.
        """      
        web.run_app(self._app, host=FACEMASK_HOST, port=FACEMASK_PORT)

