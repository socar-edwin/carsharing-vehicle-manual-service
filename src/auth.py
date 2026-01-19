"""
Authentication module for Streamlit app with Google OAuth

Uses Streamlit's built-in authentication (st.login, st.user)
with domain whitelisting for @socar.kr emails.
"""
import streamlit as st
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def check_domain_allowed(email: str, allowed_domains: list[str]) -> bool:
    """Check if email domain is in the allowed list."""
    if not email:
        return False

    email_domain = email.split("@")[-1].lower()
    return email_domain in [d.lower().strip() for d in allowed_domains]


def get_user_email() -> Optional[str]:
    """Get current user's email from Streamlit user session."""
    if not st.user.is_logged_in:
        return None
    return st.user.email


def get_user_name() -> Optional[str]:
    """Get current user's display name."""
    if not st.user.is_logged_in:
        return None
    return st.user.name


def require_auth(allowed_domains: list[str]) -> bool:
    """
    Require authentication and domain verification.

    Returns True if user is authenticated and authorized.
    Returns False and renders login UI if not.

    Usage:
        if not require_auth(["socar.kr"]):
            st.stop()
        # Continue with authenticated content
    """
    # Check if user is logged in
    if not st.user.is_logged_in:
        _render_login_page()
        return False

    # Check domain whitelist
    email = st.user.email
    if not check_domain_allowed(email, allowed_domains):
        _render_unauthorized_page(email, allowed_domains)
        return False

    logger.info(f"User authenticated: {email}")
    return True


def _render_login_page():
    """Render the login page UI."""
    st.markdown("""
    <style>
        .login-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 60vh;
            text-align: center;
        }
        .login-title {
            font-size: 2rem;
            font-weight: bold;
            color: #0078FF;
            margin-bottom: 1rem;
        }
        .login-subtitle {
            color: #666;
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<p class="login-title">차량 매뉴얼 챗봇</p>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">쏘카 계정으로 로그인해 주세요</p>', unsafe_allow_html=True)

        if st.button("Google 계정으로 로그인", type="primary", use_container_width=True):
            st.login()

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption("@socar.kr 이메일만 접근 가능합니다.")


def _render_unauthorized_page(email: str, allowed_domains: list[str]):
    """Render unauthorized access page."""
    st.error("접근 권한이 없습니다")

    st.markdown(f"""
    **로그인 이메일:** `{email}`

    이 서비스는 다음 도메인의 이메일만 사용할 수 있습니다:
    - {', '.join([f'`@{d}`' for d in allowed_domains])}

    쏘카 계정으로 다시 로그인해 주세요.
    """)

    if st.button("다른 계정으로 로그인"):
        st.logout()
        st.rerun()


def render_user_info():
    """Render user info in sidebar (optional)."""
    if st.user.is_logged_in:
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"**{st.user.name}**")
            st.caption(st.user.email)
            if st.button("로그아웃", key="logout_btn"):
                st.logout()
                st.rerun()
