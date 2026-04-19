'''K7 - 插件安装器。'''
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from server.config.settings import PluginsConfig, Settings
from server.core.plugin_manifest import find_manifest_file, parse_manifest

logger = logging.getLogger(__name__)


@dataclass
class InstallResult:
    ok: bool
    plugin_id: str = ""
    target_dir: Path = Path()
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "plugin_id": self.plugin_id,
            "target_dir": str(self.target_dir),
            "message": self.message,
        }


class PluginInstaller:
    def __init__(self, settings: Settings):
        self.settings = settings
        cfg: PluginsConfig = settings.plugins
        self.install_root = Path(cfg.install_registry).expanduser().resolve()
        self.npm_root = Path(cfg.npm_registry).expanduser().resolve()

    # ---------- 入口 ----------

    def install(self, spec: str, *, link: bool = False, force: bool = False) -> InstallResult:
        """主入口"""
        spec = (spec or "").strip()
        if not spec:
            return InstallResult(False, message="空 spec")

        try:
            if self._looks_like_local(spec):
                return self._install_local(Path(spec).expanduser().resolve(),
                                           link=link, force=force)
            if spec.startswith("git+") or spec.endswith(".git") or _looks_like_git_https(spec):
                return self._install_git(spec, force=force)
            if _looks_like_owner_repo(spec):
                url = f"https://github.com/{spec}.git"
                return self._install_git(url, force=force)
            # 其他一律走 npm
            return self._install_npm(spec, force=force)
        except Exception as e:
            logger.exception(f"install({spec}) 抛异常")
            return InstallResult(False, message=f"{type(e).__name__}: {e}")

    def uninstall(self, plugin_id: str, *, keep_files: bool = False) -> InstallResult:
        """卸载目标插件"""
        target = self.install_root / plugin_id
        if target.exists():
            if keep_files:
                return InstallResult(True, plugin_id=plugin_id, target_dir=target,
                                     message="keep_files=true，仅从 registry 记录卸载")
            shutil.rmtree(target, ignore_errors=True)
            return InstallResult(True, plugin_id=plugin_id, target_dir=target,
                                 message="目录已删除")
        # npm：尝试从 node_modules 里找
        nm = self.npm_root / "node_modules"
        for cand in (nm / "@openclaw" / plugin_id,
                     nm / plugin_id,
                     nm / f"openclaw-{plugin_id}"):
            if cand.exists():
                if keep_files:
                    return InstallResult(True, plugin_id=plugin_id,
                                         target_dir=cand,
                                         message="npm 包保留，仅 registry 卸载")
                npm_bin = shutil.which("npm")
                if not npm_bin:
                    return InstallResult(False, message="npm 不在 PATH，无法卸载 npm 包")
                rc = subprocess.run(
                    [npm_bin, "uninstall", cand.name,
                     "--prefix", str(self.npm_root)],
                    check=False, capture_output=True,
                )
                if rc.returncode != 0:
                    return InstallResult(False, message=rc.stderr.decode("utf-8", "replace"))
                return InstallResult(True, plugin_id=plugin_id, target_dir=cand,
                                     message="npm uninstall 完成")
        return InstallResult(False, plugin_id=plugin_id, message="未找到目标插件")

    # ---------- backends ----------

    def _install_local(self, src: Path, *, link: bool, force: bool) -> InstallResult:
        if not src.exists() or not src.is_dir():
            return InstallResult(False, message=f"本地路径不存在或不是目录: {src}")

        mf_file, _kind = find_manifest_file(src)
        if mf_file is None:
            return InstallResult(False,
                message=f"源目录没有 openclaw.plugin.json 或 .claude-plugin/.cursor-plugin/.codex-plugin"
            )

        manifest = parse_manifest(src)
        if manifest is None or manifest.status != "ok":
            return InstallResult(
                False, message=f"manifest 解析失败: {manifest.error if manifest else 'n/a'}")

        target = self.install_root / manifest.id
        self.install_root.mkdir(parents=True, exist_ok=True)

        if target.exists() and not force:
            return InstallResult(False, plugin_id=manifest.id,
                                 target_dir=target,
                                 message="目标已存在；使用 --force 覆盖")
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

        try:
            if link:
                target.symlink_to(src)
            else:
                shutil.copytree(src, target)
        except Exception as e:
            # 清理半成品
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            return InstallResult(False, plugin_id=manifest.id, message=str(e))

        return InstallResult(True, plugin_id=manifest.id,
                             target_dir=target,
                             message="local install 成功" + (" (link)" if link else ""))

    def _install_git(self, spec: str, *, force: bool) -> InstallResult:
        git_bin = shutil.which("git")
        if not git_bin:
            return InstallResult(False, message="git 不在 PATH，无法 clone")
        url = spec.removeprefix("git+")
        # 用目录名推断 id；manifest 解析后会重命名
        tmp_id = url.rstrip("/").split("/")[-1].removesuffix(".git") or "plugin"
        tmp_target = self.install_root / f".tmp-{tmp_id}"
        self.install_root.mkdir(parents=True, exist_ok=True)
        if tmp_target.exists():
            shutil.rmtree(tmp_target, ignore_errors=True)

        rc = subprocess.run(
            [git_bin, "clone", "--depth", "1", url, str(tmp_target)],
            check=False, capture_output=True,
        )
        if rc.returncode != 0:
            return InstallResult(False,
                message=f"git clone 失败: {rc.stderr.decode('utf-8', 'replace')}"
            )

        manifest = parse_manifest(tmp_target)
        if manifest is None or manifest.status != "ok":
            shutil.rmtree(tmp_target, ignore_errors=True)
            return InstallResult(False, message="git 仓库内无合法 manifest")

        final = self.install_root / manifest.id
        if final.exists():
            if not force:
                shutil.rmtree(tmp_target, ignore_errors=True)
                return InstallResult(False, plugin_id=manifest.id,
                                     target_dir=final,
                                     message="目标已存在；使用 --force 覆盖")
            shutil.rmtree(final, ignore_errors=True)
        tmp_target.rename(final)
        return InstallResult(True, plugin_id=manifest.id,
                             target_dir=final,
                             message="git clone 成功")

    def _install_npm(self, spec: str, *, force: bool) -> InstallResult:
        npm_bin = shutil.which("npm")
        if not npm_bin:
            return InstallResult(
                False,
                message="npm 不在 PATH。请安装 Node.js（推荐 lts），或改用本地目录/Git 安装。",
            )
        pkg = spec.removeprefix("clawhub:")
        self.npm_root.mkdir(parents=True, exist_ok=True)
        # 写空 package.json 以创建 node_modules
        pkg_json = self.npm_root / "package.json"
        if not pkg_json.exists():
            pkg_json.write_text('{"name":"openclaw-npm-root","private":true}\n',
                                encoding="utf-8")

        args = [
            npm_bin, "install", pkg,
            "--prefix", str(self.npm_root),
            "--ignore-scripts",           # 杜绝 postinstall 执行
            "--no-audit", "--no-fund",
            "--omit=dev",
        ]
        if force:
            args.append("--force")
        rc = subprocess.run(args, check=False, capture_output=True)
        if rc.returncode != 0:
            return InstallResult(
                False,
                message=f"npm install 失败: {rc.stderr.decode('utf-8', 'replace')}"
            )

        # 推断安装后目录
        nm = self.npm_root / "node_modules"
        candidates = []
        if pkg.startswith("@"):
            scope, _, name = pkg.partition("/")
            candidates.append(nm / scope / name.split("@")[0])
        else:
            candidates.append(nm / pkg.split("@")[0])
        for cand in candidates:
            if cand.exists():
                manifest = parse_manifest(cand)
                if manifest is None or manifest.status != "ok":
                    return InstallResult(
                        False,
                        message=f"npm 装好了但 {cand} 没有合法 manifest"
                    )
                return InstallResult(True, plugin_id=manifest.id,
                                     target_dir=cand,
                                     message=f"npm install 成功: {pkg}")
        return InstallResult(
            False,
            message=f"npm install 完成但找不到装入目录（pkg={pkg}）"
        )

    # ---------- helpers ----------

    @staticmethod
    def _looks_like_local(spec: str) -> bool:
        if spec.startswith("./") or spec.startswith("../") or spec.startswith("/"):
            return True
        p = Path(spec).expanduser()
        return p.exists() and p.is_dir()


def _looks_like_git_https(spec: str) -> bool:
    return (spec.startswith("https://") or spec.startswith("http://")) and spec.endswith(".git")


def _looks_like_owner_repo(spec: str) -> bool:
    if "/" not in spec:
        return False
    if spec.startswith("@") or spec.startswith("."):
        return False
    parts = spec.split("/")
    return len(parts) == 2 and all(parts) and not spec.endswith(".tgz")
