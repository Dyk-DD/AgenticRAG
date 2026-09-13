"""验证码邮件的发送。纯标准库（smtplib + email），不引入任何新依赖。

为什么是 SMTP 而不是事务邮件 API：smtplib 是标准库，零新依赖、零 DNS 改动；
用 HTTPS API 得先在本站域名上配 SPF/DKIM。容器出站 465 的 TLS 能力已实测
（smtp.qq.com / smtp.163.com / smtpdm.aliyun.com 均为 TLSv1.3），所以这条
路不需要退路。真要换通道，改本文件即可 —— 调用方只认识下面这一个函数。

配置按调用读取而不是 import 时读（同 security._secret() 的做法）：换掉 .env
重启就生效，也不必纠结 import 顺序。
"""

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr

logger = logging.getLogger(__name__)


class MailNotConfigured(RuntimeError):
    """没配 SMTP —— 部署问题，不是收件人的问题（端点映射成 503）。"""


class MailSendError(RuntimeError):
    """配好了但发不出去，或入参非法（端点映射成 502）。"""


# 主题里用的中文动作名。purpose 不认识时退回泛称，不抛异常：
# 主题文案不值得让整封信发不出去，端点那侧已经把 purpose 白名单校验过了。
_PURPOSE_LABELS = {"register": "注册", "reset": "重置密码"}


def _env(name: str, default: str = "") -> str:
    return os.getenv(name) or default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def is_configured() -> bool:
    """能不能发信。

    要求 SMTP_HOST 和 SMTP_USER 都给全，而不是只看 HOST：没有发信地址的信
    一定发不出去（AUTH 和 From 都要它），只认 HOST 会让端点放过一次注定
    失败的调用，最后以 502「发送失败」的形态呈现给用户 —— 那看起来像
    对方服务商故障，实际是本站没配好。
    """
    return bool(_env("SMTP_HOST") and _env("SMTP_USER"))


def _body(code: str, purpose: str, ttl_seconds: int) -> str:
    """纯文本正文。

    刻意不放 URL：新域名 + 纯文本信里的链接是垃圾邮件评分触发器，而这里
    没有需要链接的东西。刻意不写收件地址：正文里出现自己的邮箱是钓鱼信的
    常见特征，反而降低可信度。
    """
    minutes = max(1, ttl_seconds // 60)
    action = "注册账号" if purpose == "register" else "重置密码"
    return (
        f"你的验证码是：{code}\n\n"
        f"用于{action}，{minutes} 分钟内有效。\n"
        "如果不是你本人操作，请忽略本邮件。\n"
    )


def _login_if_needed(smtp: smtplib.SMTP, user: str, password: str) -> None:
    """有凭据才登录。

    这样 SMTP_SECURITY=none 的空配置才能指向一个本地 sink（aiosmtpd 之类）
    做免发信的演练 —— 那条路径上 AUTH 会失败在「没有凭据」而不是「服务器
    不支持」上，留下一个看不出原因的 502。
    """
    if user and password:
        smtp.login(user, password)


def send_code_email(to_email: str, code: str, purpose: str, ttl_seconds: int) -> None:
    """发一封验证码邮件。失败一律抛异常（MailNotConfigured / MailSendError）。

    同步阻塞，硬超时。**不做重试、不做服务商故障转移** —— 调用方是 FastAPI
    的同步端点，与 /api/chat 共用同一个线程池，重试会把线程池和额度一起耗光。
    """
    # 换行就是邮件头注入原语（能凭空塞一个 Bcc 出来），而 mailer 正是
    # 「字符串变成邮件头」的那道边界，所以守卫属于这里，不属于调用方。
    # 不要指望 _normalize_email 的 strip()：它只是恰好先被调到，是顺序上的
    # 巧合，不是本模块可以依赖的约束。
    if any(c in to_email for c in "\r\n\x00"):
        raise MailSendError("收件地址含非法字符")

    host = _env("SMTP_HOST")
    if not host:
        raise MailNotConfigured("未配置 SMTP_HOST")
    port = _env_int("SMTP_PORT", 465)
    user = _env("SMTP_USER")
    password = _env("SMTP_PASSWORD")
    security = _env("SMTP_SECURITY", "ssl").strip().lower()
    timeout = _env_int("SMTP_TIMEOUT_SECONDS", 10)

    msg = EmailMessage()
    msg["Subject"] = f"【智能问答系统】{_PURPOSE_LABELS.get(purpose, '')}验证码 {code}"
    msg["From"] = formataddr((_env("SMTP_FROM_NAME"), user))
    msg["To"] = to_email
    msg.set_content(_body(code, purpose, ttl_seconds), charset="utf-8")

    # save_to_sentdir 之类的开关一个都不开；更关键的是**永远不要调
    # smtp.set_debuglevel()** —— 它会打印 base64 之后的 AUTH 行，等于把
    # SMTP_PASSWORD 写进日志。debug level 必须保持 0。
    try:
        if security == "ssl":
            ctx = ssl.create_default_context()  # 证书校验打开，不降级
            with smtplib.SMTP_SSL(host, port, timeout=timeout, context=ctx) as smtp:
                _login_if_needed(smtp, user, password)
                smtp.send_message(msg)
        elif security == "starttls":
            with smtplib.SMTP(host, port, timeout=timeout) as smtp:
                smtp.starttls(context=ssl.create_default_context())
                _login_if_needed(smtp, user, password)
                smtp.send_message(msg)
        elif security == "none":
            logger.warning("SMTP_SECURITY=none：验证码邮件以明文发出，仅供本机演练")
            with smtplib.SMTP(host, port, timeout=timeout) as smtp:
                _login_if_needed(smtp, user, password)
                smtp.send_message(msg)
        else:
            raise MailNotConfigured(f"SMTP_SECURITY 取值不认识：{security}")
    except MailNotConfigured:
        raise
    except (smtplib.SMTPException, ssl.SSLError, OSError) as exc:
        # 只带类型名往上抛，完整异常交给调用方 exc_info 记日志。往上传的
        # 字符串会被塞进 HTTP 响应体，而 SMTP 的回复文本没有理由给用户看。
        raise MailSendError(type(exc).__name__) from exc
