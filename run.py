"""
یک اجراکنندهٔ واحد برای راه‌اندازی/توقف ربات و داشبورد، بدون نیاز به نوشتن دستور.

روش استفاده (VS Code):
- فایل run.py را باز کنید و کلید F5 را بزنید (یا دکمهٔ Run ▶️). برای توقف، دکمهٔ Stop ⏹️ را بزنید.
- این فایل سرور را بالا می‌آورد، مرورگر را باز می‌کند (کد سرور این کار را می‌کند)، و ربات را به‌صورت خودکار استارت می‌کند.
"""
from __future__ import annotations

import atexit
import threading
import time

import httpx
import uvicorn
import webbrowser


def _server_up(base: str = "http://127.0.0.1:8001") -> bool:
    try:
        r = httpx.get(base + "/api/status", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def _start_server():
    # اجرای uvicorn در همین فرایند؛ app همان web.server:app است
    uvicorn.run("web.server:app", host="127.0.0.1", port=8001, workers=1, reload=False)


def _wait_and_autostart_bot():
    # منتظر می‌مانیم تا سرور بالا بیاید، سپس ربات را استارت می‌کنیم
    base = "http://127.0.0.1:8001"
    for _ in range(60):  # حداکثر ~60 ثانیه صبر
        try:
            r = httpx.get(base + "/api/status", timeout=2.0)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    # استارت ربات (اگر از قبل در حال اجرا باشد، سرور خودش نادیده می‌گیرد)
    try:
        httpx.post(base + "/api/start", timeout=5.0)
    except Exception:
        pass


def _autostop_bot():
    # هنگام خروج از برنامه، ربات را متوقف می‌کنیم (ایمن)
    try:
        httpx.post("http://127.0.0.1:8001/api/stop", timeout=2.0)
    except Exception:
        pass


if __name__ == "__main__":
    base_url = "http://127.0.0.1:8001"
    if _server_up(base_url):
        # سرور در حال اجراست → فقط ربات را استارت کن و مرورگر را باز کن
        try:
            httpx.post(base_url + "/api/start", timeout=5.0)
        except Exception:
            pass
        try:
            webbrowser.open_new_tab(base_url + "/")
        except Exception:
            pass
        print("Server already running on 127.0.0.1:8001 → started bot and opened dashboard.")
    else:
        # هنگام خروج، توقف ربات را تضمین کن (فقط وقتی خودمان سرور را بالا آوردیم)
        atexit.register(_autostop_bot)
        # در یک ترد پس‌زمینه، بعد از بالا آمدن سرور ربات را استارت کن
        threading.Thread(target=_wait_and_autostart_bot, daemon=True).start()
        # سرور را اجرا کن (مرورگر به‌طور خودکار توسط web/server.py باز می‌شود)
        _start_server()
