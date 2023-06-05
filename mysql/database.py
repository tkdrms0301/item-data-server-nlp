# -*- coding: utf-8 -*-
import pymysql
from pymysql import cursors


class Database():
    def __init__(self):
        self.db = pymysql.connect(host="20.214.202.113", user="adminuser", password="1q2w3e4r", db="item", charset="utf8", port=3306, ssl={"fake_flag_to_enable_tls":True})
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
