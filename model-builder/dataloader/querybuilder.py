from typing import List, Union

class QueryBuilder:
    def __init__(self, tablename):
        self.tablename = tablename

    def get_all_columns(self) -> str:
        return "SELECT " + "* " + "FROM " + self.tablename

    def get_selected_columns(self, select_cols:List[str])-> str:
        if type(select_cols) == str:
            select_cols = [select_cols]
        if type(select_cols) == list and all([type(col) == str for col in select_cols]):
            return "SELECT " + ", ".join(select_cols) + " FROM " + self.tablename
        else:
            raise TypeError("cols must be either string of list of strings.")

    def get_all_columns_groupby(self, groupby_cols:List[str])-> str:
        if type(groupby_cols) == str:
            groupby_cols = [groupby_cols]
        if type(groupby_cols) == list and all([type(col) == str for col in groupby_cols]):
            return "SELECT " + "* " + "FROM " + self.tablename + " GROUP BY " + ", ".join(groupby_cols)
        else:
            raise TypeError("groupby_cols must be either string of list of strings.")

    def get_selected_columns_groupby(self, select_cols:List[str], groupby_cols:List[str])-> str:
        if type(select_cols) == str:
            select_cols = [select_cols]
        if type(groupby_cols) == str:
            groupby_cols = [groupby_cols]
        if type(select_cols) == list and all([type(col) == str for col in select_cols]):
            if type(groupby_cols) == list and all([type(col) == str for col in groupby_cols]):
                return "SELECT " +  ", ".join(select_cols) + " FROM " + self.tablename + " GROUP BY " + ", ".join(groupby_cols)
            else:
                raise TypeError("groupby_cols must be either string of list of strings.")
        else:
            raise TypeError("select_cols must be either string of list of strings.")

    def _check_order(self, order):
        if type(order) == str:
            if order == "ASC" or order == "DESC":
                return
            else:
                raise ValueError("Order can only be ASC or DESC")
        elif type(order) == list:
            if all([ord == "ASC" or ord == "DESC" for ord in order]):
                return
            raise ValueError("List if Order can only contain ASC or DESC")
        else:
            raise ValueError("Order must be either string of list of strings.")

    def _merge_order(self, orderby_cols:List[str], order:List[str]):
        if len(orderby_cols) != len(order):
            raise ValueError("Length of order by columns must be equal to length of order")
        return  [a + " " + b for a, b in zip(orderby_cols, order)]

    def get_all_columns_orderby(self, orderby_cols:List[str], order:List[str]):
        if type(orderby_cols) != type(order):
            raise TypeError("orderby_cols must be of same type as order.")
        self._check_order(order)
        if type(orderby_cols) == str:
            order = [order]
            orderby_cols = [orderby_cols]
        if type(orderby_cols) == list and all([type(col) == str for col in orderby_cols]):
                return "SELECT " + "* " + "FROM " + self.tablename + " ORDER BY " + ", ".join(self._merge_order(orderby_cols, order))
        else:
            raise TypeError("orderby_cols must be either string of list of strings.")

    def get_selected_columns_orderby(self, select_cols:List[str], orderby_cols:List[str], order:List[str]):
        if type(orderby_cols) != type(order):
            raise TypeError("orderby_cols must be of same type as order.")
        self._check_order(order)

        if type(select_cols) == str:
            select_cols = [select_cols]

        if type(orderby_cols) == str:
            orderby_cols = [orderby_cols]
            order = [order]

        if type(select_cols) == list and all([type(col) == str for col in select_cols]) and type(orderby_cols) == list and all([type(col) == str for col in orderby_cols]):
            return "SELECT " +  ", ".join(select_cols) + " FROM " + self.tablename + " ORDER BY " + ", ".join(self._merge_order(orderby_cols, order))
        else:
            raise TypeError("select_cols and orderby_cols must be either string of list of strings.")