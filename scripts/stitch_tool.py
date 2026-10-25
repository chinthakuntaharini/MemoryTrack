"""
Stitch MCP Client & CLI Helper
Enables direct programmatic access to the Google Stitch MCP server.
"""

import os
import sys
import json
import urllib.request
import urllib.error

STITCH_URL = "https://stitch.googleapis.com/mcp"
API_KEY = os.environ.get("STITCH_API_KEY", "")

def call_stitch(method: str, params: dict = None, req_id: int = 1):
    payload = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params or {}
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        STITCH_URL,
        data=data,
        headers={
            "X-Goog-Api-Key": API_KEY,
            "Content-Type": "application/json"
        }
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        return {"error": {"code": e.code, "message": error_body}}

def call_tool(tool_name: str, arguments: dict = None):
    return call_stitch("tools/call", {
        "name": tool_name,
        "arguments": arguments or {}
    })

def list_tools():
    return call_stitch("tools/list")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "list-tools":
            res = list_tools()
            tools = res.get("result", {}).get("tools", [])
            print(f"Discovered {len(tools)} tools:")
            for t in tools:
                print(f" - {t['name']}: {t.get('description', '').splitlines()[0]}")
        elif command == "list-projects":
            res = call_tool("list_projects")
            print(json.dumps(res, indent=2))
        elif command == "create-project":
            title = sys.argv[2] if len(sys.argv) > 2 else "New Project"
            res = call_tool("create_project", {"title": title})
            print(json.dumps(res, indent=2))
        elif command == "call" and len(sys.argv) > 2:
            tool_name = sys.argv[2]
            args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
            res = call_tool(tool_name, args)
            print(json.dumps(res, indent=2))
        else:
            print("Usage: python stitch_tool.py [list-tools | list-projects | create-project <title> | call <tool> <json-args>]")
    else:
        res = list_tools()
        tools = res.get("result", {}).get("tools", [])
        print(f"Connected to Stitch MCP. Available tools: {len(tools)}")
