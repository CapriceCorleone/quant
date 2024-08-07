import os
import sys
import cx_Oracle  # Python-Oracle连接包，需pip安装
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime


# %% 预定义参数
DB_WIND_ORACLE = {
    'user': 'rsquant',  # 数据库登录账号
    'password': 'rsQuantMszq',  # 数据库登录密码
    'dsn': '10.99.130.6:1521/wind_dg',  # 数据库DSN
}


# root = 'D:/2023MS/【基础数据】/data'  # 数据文件保存路径


# %% 数据库抓取流程
class DatabaseCollector:
    """
    Oracle数据抓取模块
    """
    # 数据库连接信息
    db_config = {'winddb': DB_WIND_ORACLE}

    # 配置
    num_rows = 100000  # 单次抓取行数上限
    roll_back = 30  # 回滚日期

    def __init__(self, root, db_name, task_list):
        self.root = root
        self.db_name = db_name
        self.task_list = task_list
        self.process_num = 1
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)

    def fetch(self, sql):
        # build connection
        conn = cx_Oracle.connect(**self.db_config[self.db_name])
        print("111")
        cur = conn.cursor()

        # execute query
        cur.execute(sql)
        columns = [i[0] for i in cur.description]  # column names
        clob_field = [i[0] for i in cur.description if i[1].name == 'DB_TYPE_CLOB']  # clob columns names

        # fetch data
        fetched = []
        fetched_rows = 0
        while True:
            rows = cur.fetchmany(self.num_rows)
            # fetched.extend(rows)
            if not rows:
                break
            else:
                rows = pd.DataFrame(rows, columns=columns)
                # if clob columns do read
                if len(clob_field) > 0:
                    rows[clob_field] = rows[clob_field].applymap(lambda x: x.read() if x is not None else None)
                fetched_rows += len(rows)
                fetched.append(rows)
                sys.stdout.write("\r Fetched rows: " + str(fetched_rows))  # display number of fetched rows
                sys.stdout.flush()

        conn.close()
        sys.stdout.write("\n")
        return pd.concat(fetched, axis=0)

    @staticmethod
    def convert_64_to_32(data):
        # force convert 64 digit to 32 digit
        conversion = {
            'float64': np.float32,
            'int64': np.int32
        }

        for typ in conversion:
            field = data.dtypes.loc[data.dtypes == typ].index.values
            if len(field) > 0:
                data[field] = data[field].astype(conversion[typ])
        return data

    @staticmethod
    def convert_object_to_category(data, n=10):

        # infer category dtype
        object_field = data.dtypes.loc[data.dtypes == 'object'].index
        unique_num = data[object_field].nunique()
        category_field = unique_num.loc[unique_num <= n].index

        # convert
        if len(category_field) > 0:
            data[category_field] = data[category_field].astype('category')
        return data

    @staticmethod
    def save(data, path):
        data.to_pickle(path, compression=None)

    @staticmethod
    def correct_date(data, date_field='ANN_DT'):
        opdate = data.OPDATE.copy()
        opdate.loc[opdate.apply(lambda x: x.hour) < 5] = opdate.loc[
                                                             opdate.apply(lambda x: x.hour) < 5] - pd.to_timedelta(1,
                                                                                                                   unit='D')
        opdate = opdate.apply(lambda x: x.strftime('%Y%m%d'))
        ix_push = (data[date_field] == opdate)
        data_ = data.copy()
        data_.loc[ix_push, date_field] = (
                pd.to_datetime(data.loc[ix_push, date_field]) + pd.to_timedelta(1, unit='D')).apply(
            lambda x: x.strftime('%Y%m%d'))
        return data_

    def exec_task(self, task):
        table = task.__name__
        print(f'{datetime.now()}: pid {os.getpid()} -> Processing table: [{table}]')

        path = f'{self.root}/{table}.pkl'

        incremental = True if task.task == 'incremental' else False
        reindex = True if task.field_index is not None else False

        init_date = None
        if incremental:
            if os.path.exists(path):
                file_existed = pd.read_pickle(path)
                if reindex:
                    if isinstance(file_existed.index, pd.MultiIndex):
                        file_date = file_existed.index.levels[0].max()
                    else:
                        file_date = file_existed.index.max()
                    init_date = (pd.to_datetime(file_date) - pd.to_timedelta(self.roll_back, unit='D')).strftime(
                        '%Y%m%d')
                    file_existed = file_existed.loc[:init_date]
                else:
                    file_date = file_existed[task.field_date].max()
                    init_date = (pd.to_datetime(file_date) - pd.to_timedelta(self.roll_back, unit='D')).strftime(
                        '%Y%m%d')
                    file_existed = file_existed.loc[file_existed[task.field_date] <= init_date]

            else:
                init_date = 20041231

        fetched = self.fetch(task.query(init_date))

        if incremental:
            print(" Last date: " + str(fetched[task.field_date].max()))

        for col in task.field_omit:
            if col in fetched.columns:
                del fetched[col]

        if reindex:
            fetched = fetched.set_index(task.field_index).sort_index()

        fetched = self.convert_64_to_32(fetched)
        fetched = self.convert_object_to_category(fetched)

        if incremental:
            if os.path.exists(path):
                fetched = pd.concat([file_existed, fetched])

        if reindex:
            fetched = fetched.loc[~fetched.index.duplicated(keep='last')]
        else:
            fetched = fetched.drop_duplicates(keep='last').reset_index(drop=True)

        self.save(fetched, path)

    def run(self):
        print(f'\nCollecting data from [{self.db_name}] to {self.root} with {self.process_num} processes...')

        if self.process_num > 1:
            pool = mp.Pool(self.process_num)

        for task in self.task_list:
            try:
                if self.process_num > 1:
                    pool.apply_async(self.exec_task, (task,))
                else:
                    self.exec_task(task)
            except:
                import traceback
                print(traceback.print_exc())
                print(f'{datetime.now()}: Error at: [{task.__name__}]')

        if self.process_num > 1:
            pool.close()
            pool.join()