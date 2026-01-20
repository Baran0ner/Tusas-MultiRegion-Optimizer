import os

from flask import Flask


def create_app() -> Flask:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    app = Flask(
        __name__,
        static_folder=os.path.join(root_dir, "static"),
        static_url_path="/static",
        template_folder=os.path.join(root_dir, "templates"),
    )

    from .api.routes import bp

    app.register_blueprint(bp)
    return app

