from __future__ import annotations

import glob as glob_module
import subprocess
from pathlib import Path

from core.tool import Tool, ToolResult

_MAX_RESULTS = 100


class GlobTool(Tool):
    name = "Glob"
    description = (
        "- Fast file pattern matching tool\n"
        "- Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\"\n"
        "- Returns matching file paths sorted by modification date\n"
        "- Use when you need to find files by name patterns"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern for file matching"},
            "path": {"type": "string", "description": "Directory to search in (default: cwd)"},
        },
        "required": ["pattern"],
    }

    def is_read_only(self) -> bool:
        return True

    def get_activity_description(self, **kwargs) -> str | None:
        pattern = kwargs.get("pattern", "")
        return f"Finding {pattern}" if pattern else None

    def execute(self, pattern: str, path: str = ".") -> ToolResult:
        base = Path(path).resolve()
        if not base.exists():
            return ToolResult(content=f"Error: Directory not found: {path}", is_error=True)
        if not base.is_dir():
            return ToolResult(content=f"Error: Path is not a directory: {path}", is_error=True)

        try:
            matches = self._rg_glob(pattern, str(base))
        except FileNotFoundError:
            matches = self._python_glob(pattern, base)

        if not matches:
            return ToolResult(content="No files found matching the pattern.")

        truncated = len(matches) > _MAX_RESULTS
        matches = matches[:_MAX_RESULTS]

        rel_matches = []
        for m in matches:
            try:
                rel_matches.append(str(Path(m).relative_to(base)))
            except ValueError:
                rel_matches.append(m)

        result = "\n".join(rel_matches)
        if truncated:
            result += "\n(Results are truncated. Consider using a more specific path or pattern.)"
        return ToolResult(content=result)

    def _rg_glob(self, pattern: str, search_dir: str) -> list[str]:
        cmd = ["rg", "--files", "--glob", pattern, "--sort=modified",
               "--no-ignore", "--hidden", search_dir]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )
        if result.returncode not in (0, 1):
            raise FileNotFoundError("rg failed")
        output = result.stdout.strip()
        return output.split("\n") if output else []

    def _python_glob(self, pattern: str, base: Path) -> list[str]:
        matches = glob_module.glob(pattern, root_dir=str(base), recursive=True)
        matches = sorted(matches, key=lambda p: (base / p).stat().st_mtime, reverse=True)
        return [str(base / m) for m in matches]
