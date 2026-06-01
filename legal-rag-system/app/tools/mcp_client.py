"""MCP 客户端管理器 — 连接 MCP 服务端 (SSE 传输)，发现和调用远程工具"""
import asyncio
import json
import os
import sys
import subprocess
import threading
from typing import Dict, List, Any, Optional


class MCPTool:
    """通过 MCP 协议发现的远程工具包装器"""
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any],
                 server_name: str, client_manager: "MCPClientManager",
                 remote_name: str = None):
        self.name = name                    # 注册到 ToolRegistry 的名称 (如 mcp_fetch)
        self.remote_name = remote_name or name  # MCP 服务端的原始工具名 (如 fetch)
        self.description = description
        self.source = "mcp"
        self.input_schema = input_schema
        self.server_name = server_name
        self._manager = client_manager

    def execute(self, input_text: str = "", **kwargs) -> str:
        """同步执行 MCP 工具（桥接 async → sync）

        将 execute_tool 传入的 input/query 等通用参数名映射到 MCP 工具 schema 实际要求的参数名。
        同时处理 LLM 传递的 JSON 对象格式（如 {"url": "https://..."}）。
        """
        raw_input = input_text or kwargs.pop("input", "") or kwargs.pop("query", "") or ""

        # Strip surrounding matched quotes only (LLM may wrap entire input: "https://..." or 'https://...')
        # Do NOT blindly strip("\"'") — it corrupts key=value syntax like query="foo"
        if isinstance(raw_input, str) and len(raw_input) >= 2:
            if (raw_input[0] == raw_input[-1] == '"') or (raw_input[0] == raw_input[-1] == "'"):
                raw_input = raw_input[1:-1]

        props = self.input_schema.get("properties", {})
        mapped = {}

        # 尝试解析 JSON（LLM 可能直接传 JSON 对象字符串）
        if isinstance(raw_input, str) and raw_input.strip().startswith("{"):
            try:
                parsed = json.loads(raw_input)
                if isinstance(parsed, dict):
                    for key, val in parsed.items():
                        if key in props:
                            mapped[key] = val
                    if not mapped:
                        mapped = parsed
            except (json.JSONDecodeError, Exception):
                pass

        # 尝试解析 key=value 或 key: value 格式（ReAct agent 传参: query="xxx", max_results=5）
        if not mapped and isinstance(raw_input, str) and ("=" in raw_input or ":" in raw_input):
            import re as _re_mcp
            # 判断分隔符类型：优先 = ，其次 :
            sep = "=" if "=" in raw_input else ":"
            # Strategy: split by ", KEY<sep>" pattern to handle multi-word unquoted values
            # First try quoted values, then split on ", \w+<sep>" boundaries
            remaining = raw_input
            while remaining:
                m = _re_mcp.match(r'(\w+)\s*' + _re_mcp.escape(sep) + r'\s*(?:"([^"]*)"|\'([^\']*)\')', remaining)
                if m:
                    key = m.group(1)
                    val = m.group(2) or m.group(3)
                    remaining = remaining[m.end():].lstrip(',， \t')
                else:
                    # Unquoted value: match key<sep>val where val stops at ", KEY<sep>" or end
                    m = _re_mcp.match(r'(\w+)\s*' + _re_mcp.escape(sep) + r'\s*(.+?)(?=$|,\s*\w+\s*[=:])', remaining)
                    if m:
                        key = m.group(1)
                        val = m.group(2).strip()
                        remaining = remaining[m.end():].lstrip(',， \t')
                    else:
                        break
                if val and isinstance(val, str) and val.isdigit():
                    val = int(val)
                if key in props:
                    mapped[key] = val
                elif not mapped:
                    mapped[key] = val

        # 回退：取第一个 string 类型参数作为 key
        if not mapped and raw_input:
            for key, val in props.items():
                if val.get("type") == "string":
                    mapped[key] = raw_input
                    break

        if not mapped and raw_input:
            mapped["input"] = raw_input

        return self._manager.call_tool_sync(self.server_name, self.remote_name, mapped)


class MCPClientManager:
    """管理 MCP 服务端连接，使用 SSE 传输避免 stdio/anyio 线程兼容性问题"""

    def __init__(self):
        self._servers: Dict[str, dict] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._running = False

    def add_sse_server(self, name: str, url: str):
        """注册一个 SSE 传输的 MCP 服务端"""
        self._servers[name] = {
            "transport": "sse",
            "url": url,
            "session": None,
            "_session_ctx": None,
            "_sse_ctx": None,
            "tools": [],
        }

    def add_streamable_http_server(self, name: str, url: str):
        """注册一个 Streamable HTTP 传输的远程 MCP 服务端 (如 ModelScope)"""
        self._servers[name] = {
            "transport": "streamable_http",
            "url": url,
            "session": None,
            "_session_ctx": None,
            "_http_ctx": None,
            "tools": [],
        }

    def add_local_server(self, name: str, port: int = 8765, module: str = "app.mcp.legal_mcp_server"):
        """注册并通过子进程启动本地 MCP 服务端 (SSE 模式)

        Args:
            name: 服务名称
            port: 监听端口
            module: Python 模块路径，如 app.mcp.legal_mcp_server 或 app.mcp.fetch_mcp_server
        """
        self._servers[name] = {
            "transport": "sse_local",
            "port": port,
            "url": f"http://127.0.0.1:{port}/sse",
            "module": module,
            "process": None,
            "session": None,
            "_session_ctx": None,
            "tools": [],
        }

    def start(self, timeout: float = 15.0):
        """在后台线程启动事件循环并连接所有 MCP 服务端"""
        self._ready.clear()
        self._running = True

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._connect_all())
            except Exception as e:
                print(f"[MCP] 连接失败: {e}")
            self._ready.set()
            while self._running:
                try:
                    self._loop.run_forever()
                except Exception:
                    break

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            print(f"[MCP] 连接超时 ({timeout}s)，将以无 MCP 工具模式运行")

    async def _connect_all(self):
        """连接所有已注册的 MCP 服务端"""
        try:
            from mcp.client.sse import sse_client
        except ImportError:
            print("[MCP] mcp 包未安装，跳过 MCP 工具加载")
            return

        for name, cfg in self._servers.items():
            try:
                if cfg["transport"] == "sse_local":
                    await self._connect_local_sse(name, cfg, sse_client)
                elif cfg["transport"] == "sse":
                    await self._connect_sse(name, cfg, sse_client)
                elif cfg["transport"] == "streamable_http":
                    await self._connect_streamable_http(name, cfg)
            except asyncio.TimeoutError:
                print(f"[MCP] ✗ 连接 '{name}' 超时")
            except Exception as e:
                print(f"[MCP] ✗ 连接 '{name}' 失败: {e}")

    async def _connect_local_sse(self, name: str, cfg: dict, sse_client):
        """启动本地 MCP 服务端子进程并通过 SSE 连接（含重试）"""
        port = cfg["port"]

        module = cfg.get("module", "app.mcp.legal_mcp_server")

        # Launch the server as a subprocess
        env = os.environ.copy()
        process = subprocess.Popen(
            [sys.executable, "-m", module, "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        cfg["process"] = process

        # Retry connection up to 5 times (server needs time to start)
        last_err = None
        for attempt in range(5):
            await asyncio.sleep(1.0)
            try:
                await self._connect_sse(name, cfg, sse_client)
                return  # Success
            except Exception as e:
                last_err = e
                # Clean up any partial context from failed attempt
                await self._safe_close_ctx(cfg, "_session_ctx")
                await self._safe_close_ctx(cfg, "_sse_ctx")

        print(f"[MCP] ✗ 连接 '{name}' 失败 (重试5次): {last_err}")

    async def _safe_close_ctx(self, cfg: dict, key: str):
        """安全关闭 anyio context manager，忽略跨任务清理错误"""
        ctx = cfg.pop(key, None)
        if ctx:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass

    async def _connect_sse(self, name: str, cfg: dict, sse_client):
        """通过 SSE 连接到已有 MCP 服务端"""
        url = cfg["url"]
        ctx = sse_client(url)
        read, write = await asyncio.wait_for(ctx.__aenter__(), timeout=10.0)
        cfg["_sse_ctx"] = ctx

        from mcp import ClientSession
        session_ctx = ClientSession(read, write)
        session = await asyncio.wait_for(session_ctx.__aenter__(), timeout=5.0)
        cfg["_session_ctx"] = session_ctx
        cfg["session"] = session

        await asyncio.wait_for(session.initialize(), timeout=5.0)

        result = await asyncio.wait_for(session.list_tools(), timeout=5.0)
        tools = []
        for tool in result.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
            })

        cfg["tools"] = tools
        print(f"[MCP] ✓ 已连接 '{name}' (SSE) — 发现 {len(tools)} 个工具: "
              f"{', '.join(t['name'] for t in tools)}")

    async def _connect_streamable_http(self, name: str, cfg: dict):
        """通过 Streamable HTTP 连接到远程 MCP 服务端 (如 ModelScope)"""
        from mcp.client.streamable_http import streamablehttp_client
        url = cfg["url"]

        ctx = streamablehttp_client(url)
        read, write, get_session_id = await asyncio.wait_for(ctx.__aenter__(), timeout=15.0)
        cfg["_http_ctx"] = ctx

        from mcp import ClientSession
        session_ctx = ClientSession(read, write)
        session = await asyncio.wait_for(session_ctx.__aenter__(), timeout=10.0)
        cfg["_session_ctx"] = session_ctx
        cfg["session"] = session

        await asyncio.wait_for(session.initialize(), timeout=10.0)

        result = await asyncio.wait_for(session.list_tools(), timeout=10.0)
        tools = []
        for tool in result.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
            })

        cfg["tools"] = tools
        print(f"[MCP] ✓ 已连接 '{name}' (StreamableHTTP) — 发现 {len(tools)} 个工具: "
              f"{', '.join(t['name'] for t in tools)}")

    def list_all_tools(self) -> List[Dict[str, Any]]:
        """列出所有 MCP 服务端发现的工具"""
        tools = []
        for name, cfg in self._servers.items():
            for tool in cfg.get("tools", []):
                tools.append({**tool, "server": name})
        return tools

    def get_mcp_tools(self) -> List[MCPTool]:
        """返回可注册到 ToolRegistry 的 MCPTool 列表"""
        result = []
        for name, cfg in self._servers.items():
            for tool in cfg.get("tools", []):
                result.append(MCPTool(
                    name=f"mcp_{tool['name']}",
                    remote_name=tool["name"],
                    description=f"[MCP:{name}] {tool.get('description', '')}",
                    input_schema=tool.get("input_schema", {}),
                    server_name=name,
                    client_manager=self,
                ))
        return result

    def call_tool_sync(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """同步调用 MCP 工具"""
        if not self._loop or not self._loop.is_running():
            return f"[MCP] 错误: 事件循环未运行"

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._call_tool(server_name, tool_name, arguments),
                self._loop,
            )
            return future.result(timeout=30)
        except asyncio.TimeoutError:
            return f"[MCP] 错误: 调用 '{tool_name}' 超时"
        except Exception as e:
            return f"[MCP] 错误: 调用 '{tool_name}' 失败: {e}"

    async def _call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """异步调用 MCP 工具"""
        cfg = self._servers.get(server_name)
        if not cfg or not cfg.get("session"):
            return f"[MCP] 服务 '{server_name}' 未连接"

        try:
            result = await asyncio.wait_for(
                cfg["session"].call_tool(tool_name, arguments),
                timeout=20.0,
            )
            if hasattr(result, 'content') and result.content:
                parts = []
                for c in result.content:
                    if hasattr(c, 'text'):
                        parts.append(c.text)
                return "\n".join(parts) if parts else str(result)
            return str(result)
        except Exception as e:
            import traceback
            detail = str(e) or type(e).__name__
            return f"[MCP] 工具 '{tool_name}' 执行失败: {detail}\nargs: {json.dumps(arguments, ensure_ascii=False)}"

    def shutdown(self):
        """关闭所有 MCP 连接"""
        self._running = False

        async def _close():
            for cfg in self._servers.values():
                try:
                    ctx = cfg.get("_session_ctx")
                    if ctx:
                        await ctx.__aexit__(None, None, None)
                except Exception:
                    pass
                for transport_key in ("_sse_ctx", "_http_ctx"):
                    try:
                        ctx = cfg.get(transport_key)
                        if ctx:
                            await ctx.__aexit__(None, None, None)
                    except Exception:
                        pass
                try:
                    process = cfg.get("process")
                    if process:
                        process.kill()
                        process.wait(timeout=3)
                except Exception:
                    pass

        if self._loop and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(_close(), self._loop)
                future.result(timeout=5)
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass


_mcp_manager: Optional[MCPClientManager] = None


def get_mcp_manager() -> MCPClientManager:
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPClientManager()
    return _mcp_manager
