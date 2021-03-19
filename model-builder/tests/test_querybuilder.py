import unittest
from dataloader.querybuilder import QueryBuilder

class TestQueryBuilder(unittest.TestCase):
    def setUp(self):
        self.tablename = "test"
        self.querybuilder = QueryBuilder(self.tablename)

    def test_get_all_columns(self):
        response = self.querybuilder.get_all_columns()
        expected = "SELECT * FROM test"
        self.assertEqual(response, expected)

    def test_get_selected_columns_string_input(self):
        response = self.querybuilder.get_selected_columns("column1")
        expected = "SELECT column1 FROM test"
        self.assertEqual(response, expected)

    def test_get_selected_columns_list_of_str_input(self):
        response = self.querybuilder.get_selected_columns(["column1", "column2"])
        expected = "SELECT column1, column2 FROM test"
        self.assertEqual(response, expected)

    def test_get_selected_columns_int_input(self):
        self.assertRaises(TypeError, self.querybuilder.get_selected_columns, 1)

    def test_get_selected_columns_list_of_int_input(self):
        self.assertRaises(TypeError, self.querybuilder.get_selected_columns, [1,2])

    def test_get_selected_columns_list_of_various_input(self):
        self.assertRaises(TypeError, self.querybuilder.get_selected_columns, ["col1", 1, 2])

    def test_get_all_columns_groupby_str_input(self):
        response = self.querybuilder.get_all_columns_groupby("column1")
        expected = "SELECT * FROM test GROUP BY column1"
        self.assertEqual(response, expected)

    def test_get_all_columns_groupby_list_of_str_input(self):
        response = self.querybuilder.get_all_columns_groupby(["column1", "column2"])
        expected = "SELECT * FROM test GROUP BY column1, column2"
        self.assertEqual(response, expected)

    def test_get_all_columns_groupby_int_input(self):
        self.assertRaises(TypeError, self.querybuilder.get_selected_columns_groupby, 42)

    def test_get_all_columns_list_of_various_input(self):
        self.assertRaises(TypeError, self.querybuilder.get_selected_columns_groupby, ["column1", 42, 32])

    def test_get_selected_columns_groupby_str_str_input(self):
        response = self.querybuilder.get_selected_columns_groupby("column1", "column2")
        expected = "SELECT column1 FROM test GROUP BY column2"
        self.assertEqual(response, expected)

    def test_get_selected_columns_groupby_list_str_input(self):
        response = self.querybuilder.get_selected_columns_groupby(["column1", "column2"], "column3")
        expected = "SELECT column1, column2 FROM test GROUP BY column3"
        self.assertEqual(response, expected)

    def test_get_selected_columns_groupby_str_list_input(self):
        response = self.querybuilder.get_selected_columns_groupby("column1", ["column2", "column3"])
        expected = "SELECT column1 FROM test GROUP BY column2, column3"
        self.assertEqual(response, expected)

    def test_get_selected_columns_groupby_list_list_input(self):
        response = self.querybuilder.get_selected_columns_groupby(["column1", "column2"], ["column3", "column4"])
        expected = "SELECT column1, column2 FROM test GROUP BY column3, column4"
        self.assertEqual(response, expected)

    def test_get_all_columns_orderby_str_input(self):
        response = self.querybuilder.get_all_columns_orderby("column1", "ASC")
        expected = "SELECT * FROM test ORDER BY column1 ASC"
        self.assertEqual(response, expected)

    def test_get_all_columns_orderby_list_of_str_input(self):
        response = self.querybuilder.get_all_columns_orderby(["column1", "column2"], ["ASC", "DESC"])
        expected = "SELECT * FROM test ORDER BY column1 ASC, column2 DESC"
        self.assertEqual(response, expected)

    def test_get_all_columns_orderby_int_input(self):
        self.assertRaises(TypeError, self.querybuilder.get_selected_columns_orderby, 42, "ASC")

    def test_get_all_columns_orderby_list_of_various_input(self):
        self.assertRaises(TypeError, self.querybuilder.get_selected_columns_orderby, ["column1", 42, 32], ["ASC", "ASC", "ASC"])

    def test_get_all_columns_orderby_list_of_various_input(self):
        self.assertRaises(TypeError, self.querybuilder.get_selected_columns_orderby, ["column1", 42, 32], ["ASC", "ASC", "ASC"])

    def test_get_all_columns_orderby_list_of_various_order_input(self):
        self.assertRaises(TypeError, self.querybuilder.get_selected_columns_orderby, ["column1", "column2"], ["ASC", "42"])

    def test_get_all_columns_orderby_different_type_input(self):
        self.assertRaises(TypeError, self.querybuilder.get_selected_columns_orderby, ["column1", "column2"], "ASC")

    def test_get_all_columns_orderby_different_len_input(self):
        self.assertRaises(TypeError, self.querybuilder.get_selected_columns_orderby, ["column1", "column2"], ["ASC", "DESC", "ASC"])

    def test_get_selected_columns_orderby_str_str_input(self):
        response = self.querybuilder.get_selected_columns_orderby("column1", "column2", "DESC")
        expected = "SELECT column1 FROM test ORDER BY column2 DESC"
        self.assertEqual(response, expected)

    def test_get_selected_columns_orderby_list_str_input(self):
        response = self.querybuilder.get_selected_columns_orderby(["column1", "column2"], "column3", "DESC")
        expected = "SELECT column1, column2 FROM test ORDER BY column3 DESC"
        self.assertEqual(response, expected)

    def test_get_selected_columns_orderby_str_list_input(self):
        response = self.querybuilder.get_selected_columns_orderby("column1", ["column2", "column3"], ["ASC", "DESC"])
        expected = "SELECT column1 FROM test ORDER BY column2 ASC, column3 DESC"
        self.assertEqual(response, expected)

    def test_get_selected_columns_orderby_list_list_input(self):
        response = self.querybuilder.get_selected_columns_orderby(["column1", "column2"], ["column3", "column4"], ["ASC", "DESC"])
        expected = "SELECT column1, column2 FROM test ORDER BY column3 ASC, column4 DESC"
        self.assertEqual(response, expected)