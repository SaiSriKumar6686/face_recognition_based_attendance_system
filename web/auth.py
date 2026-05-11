"""
auth.py
───────
Flask-Login integration for admin authentication.

Provides:
  - User loader for Flask-Login
  - Login/logout routes
  - @login_required decorator for sensitive routes
  - Default admin account (admin/admin123) created on first run
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

from src.utils.db import get_session, AdminUser, log_audit

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message_category = "warning"

auth_bp = Blueprint("auth", __name__)


class AuthUser(UserMixin):
    """Wrapper for Flask-Login compatibility."""
    def __init__(self, admin_user: AdminUser):
        self.id = admin_user.id
        self.username = admin_user.username
        self.role = admin_user.role


@login_manager.user_loader
def load_user(user_id):
    with get_session() as s:
        admin = s.query(AdminUser).get(int(user_id))
        if admin:
            return AuthUser(admin)
    return None


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with get_session() as s:
            admin = s.query(AdminUser).filter_by(username=username).first()
            if admin and admin.check_password(password):
                user = AuthUser(admin)
                login_user(user, remember=True)
                log_audit(username, "LOGIN", details="Successful login")
                next_page = request.args.get("next")
                return redirect(next_page or url_for("dashboard"))
            else:
                error = "Invalid username or password"
                log_audit(username or "unknown", "LOGIN_FAILED", details="Invalid credentials")

    return render_template("login.html", error=error)


@auth_bp.route("/logout")
@login_required
def logout():
    log_audit(current_user.username, "LOGOUT")
    logout_user()
    return redirect(url_for("auth.login"))
