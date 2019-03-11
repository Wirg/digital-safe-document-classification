import sqlite3
import io
import numpy as np

DEFAULT_DATABASE = 'data/app.database.sqlite'


def array_to_sqlite_binary(array):
    stream = io.BytesIO()
    np.save(stream, array)
    stream.seek(0)
    return sqlite3.Binary(stream.read())


def sqlite_binary_to_array(text):
    stream = io.BytesIO(text)
    stream.seek(0)
    return np.load(stream)


sqlite3.register_adapter(np.ndarray, array_to_sqlite_binary)
sqlite3.register_converter("array", sqlite_binary_to_array)


class DataBaseCursor:
    def __init__(self, filename=DEFAULT_DATABASE):
        self._filename = filename
        self._conn = None
        self._curs = None

    @property
    def cursor(self):
        if self._curs is None:
            self._conn = sqlite3.connect(
                self._filename,
                detect_types=sqlite3.PARSE_DECLTYPES,  # Adding array type
                isolation_level=None,  # Autocommit
            )
            self._curs = self._conn.cursor()
        return self._curs

    def close(self):
        self._conn.commit()
        self._conn.close()
        self._conn = None
        self._curs = None

    def __enter__(self):
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_table_document(db_name=DEFAULT_DATABASE):
    with DataBaseCursor(db_name) as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS document (
        name VARCHAR, folder VARCHAR, content TEXT, vector array, userid INT,
        UNIQUE(name, folder, userid) ON CONFLICT REPLACE
        )""")


def add_document(user_id, filename, content, vector, folder=None, db_name=DEFAULT_DATABASE):
    with DataBaseCursor(db_name) as cur:
        cur.execute(
            'INSERT INTO document (name, folder, content, vector, userid) values (?, ?, ?, ?, ?)',
            (filename, folder, content, vector, user_id)
        )
        return cur.lastrowid


def find_user_documents(user_id, db_name=DEFAULT_DATABASE):
    with DataBaseCursor(db_name) as cur:
        cur.execute('SELECT folder, vector, name, content FROM document WHERE userid=?', (user_id,))
        return cur.fetchall()


def find_user_folder_representation(user_id, db_name=DEFAULT_DATABASE):
    with DataBaseCursor(db_name) as cur:
        cur.execute(
            'SELECT folder, vector FROM document WHERE userid=? AND folder IS NOT NULL AND vector IS NOT NULL',
            (user_id,)
        )
        user_data = cur.fetchall()
    if not user_data:
        return [], []
    folders, vectors = zip(*user_data)
    return list(folders), np.concatenate(vectors)


def change_document_folder(row_id, folder, db_name=DEFAULT_DATABASE):
    with DataBaseCursor(db_name) as cur:
        cur.execute("""
        UPDATE document
        SET folder = ?
        WHERE rowid = ?
        """, (folder, row_id))
