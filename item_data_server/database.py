# -*- coding: utf-8 -*-
import pymysql
from pymysql import cursors


class Database():
    def __init__(self):
        self.db = pymysql.connect(host="{host}", user="{user}", password="{password}", db="{db}", charset="utf8", port={port}, ssl={"fake_flag_to_enable_tls":True})
        self.cursor = self.db.cursor(cursors.DictCursor)

    def execute(self, query):
        self.cursor.execute(query)

    def execute_one(self, query):
        self.cursor.execute(query)
        row = self.cursor.fetchone()
        return row

    def execute_all(self, query):
        self.cursor.execute(query)
        row = self.cursor.fetchall()
        return row

    def commit(self):
        self.db.commit()
