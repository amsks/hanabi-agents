import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declaritive_base
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey
import pickle

Base = declaritive_base()
class DataBase:

    def __init__(self, path_to_db):
        self.engine = create_engine('sqlite://{path_to_db}')
        self.base
    

    def create_connection(self, path):
        """ create a database connection to the SQLite database
                specified by path
            :param path: database file
            :return: Connection object or None
        """
        try:
            conn = sqlite3.connect(str(path))
            return conn
        except sqlite3.Error as e:
            print(e)
        return None



    def add_entry(self, observation):
        pass

    def retrieve_entry(self, type = None, list_id = None):
        pass

    class ObsObject(Base):
        __tablename__ = 'ObsObject'
        __table_args__ = {'sqlite_autoincrement': True}

        id = Column(Integer, primary_key = True)
        current_player_offset = Column(Integer)
        hands = Column(String)
        discard_pile = Column(String)
        fireworks = Column(String)


    

