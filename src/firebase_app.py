"""Firebase app initialization helpers."""

import firebase_admin
from firebase_admin import credentials


def initialize_firebase_app(
    firebase_creds_path: str,
    firebase_db_url: str,
) -> firebase_admin.App:
    """Initialize and return the default Firebase app."""
    try:
        app = firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(firebase_creds_path)
        return firebase_admin.initialize_app(
            cred,
            {"databaseURL": firebase_db_url},
        )

    database_url = app.options.get("databaseURL")
    if not database_url:
        raise ValueError("Firebase app is initialized without a databaseURL")
    if database_url != firebase_db_url:
        raise ValueError(
            "Firebase app is already initialized with a different databaseURL"
        )
    return app