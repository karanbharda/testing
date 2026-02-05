#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()
import os, json, traceback
from pathlib import Path

print('Current working dir:', Path.cwd())
print('DHAN_CLIENT_ID:', os.getenv('DHAN_CLIENT_ID'))
print('DHAN_ACCESS_TOKEN:', 'SET' if os.getenv('DHAN_ACCESS_TOKEN') else 'MISSING')

try:
    from backend.dhan_client import DhanAPIClient
    c = os.getenv('DHAN_CLIENT_ID')
    t = os.getenv('DHAN_ACCESS_TOKEN')
    client = DhanAPIClient(client_id=c, access_token=t)
    resp = client.get_funds()
    print('Dhan get_funds response:')
    print(json.dumps(resp, indent=2, default=str))
except Exception as e:
    print('Error calling Dhan client:')
    traceback.print_exc()
