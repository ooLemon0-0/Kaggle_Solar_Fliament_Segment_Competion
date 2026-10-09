import _bootstrap
from download_route_assets import fetch
from src.utils.common import ROOT
fetch('https://huggingface.co/mouseland/cellpose-sam/resolve/7c61431b5fbb078f3296754bd15d9f51b320f837/cpdino-vitb',ROOT/'output/pretrained/routes_v3/cellpose/cpdino-vitb',
      expected_sha256='3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f')
