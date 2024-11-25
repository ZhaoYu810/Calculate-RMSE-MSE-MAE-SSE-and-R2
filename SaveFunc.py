import numpy as np
import pandas as pd
import os


def write_data_to_excel_col(data, file_path, column_name):
    """
    将字典、列表或元组的数据写入到Excel文件中。

    :param data: 要写入的数据，可以是字典、列表或元组。
    :param file_path: Excel文件的路径。
    注意：输入为array时会被展平后保存
    """
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
        new_data = []

        if isinstance(data, dict):
            new_data = [data.get(key, None) for key in data]
        elif isinstance(data, list):
            new_data = data
        elif isinstance(data, tuple):
            new_data = list(data)
        elif isinstance(data, np.ndarray):
            new_data = data.flatten().tolist()

        new_df = pd.DataFrame({
            column_name: new_data
        })

        combined_df = pd.concat([existing_df, new_df], axis=1)
        combined_df.to_excel(file_path, index=False)
    else:
        if isinstance(data, dict):
            df = pd.DataFrame({
                column_name: [data.get(key, None) for key in data]
            })
        elif isinstance(data, list):
            df = pd.DataFrame({
                column_name: data
            })
        elif isinstance(data, tuple):
            df = pd.DataFrame({
                column_name: list(data)
            })

        df.to_excel(file_path, index=False)

def write_data_to_excel_row(data, file_path):
    """
    将字典、列表或元组的数据写入到Excel文件中。

    :param data: 要写入的数据，可以是字典、列表或元组。
    :param file_path: Excel文件的路径。
    """
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
    else:
        existing_df = pd.DataFrame()

    new_data = []

    if isinstance(data, dict):
        new_data = list(data.items())
    elif isinstance(data, list):
        new_data = [(item,) for item in data]
    elif isinstance(data, tuple):
        new_data = [data]

    new_df = pd.DataFrame(new_data)

    # combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = pd.concat([existing_df, new_df], axis=1)
    combined_df.to_excel(file_path, index=False)

if __name__ == "__main__":
    save_file_path = "D:\\Pycharm\\project\\reformExcelData\\OceanOil\\check\\saveFuncTest\\test.xlsx"
    dic_example = {'Alice':20, 'Phoenix':24, 'Bob':12, 'Charlie':19}
    list_example = [1, 2, 3, 4]
    Tuple_example = (1, 2, 3, "apple", "banana", "cherry", True, False, 3.14, 2.718, (4, 5, 6), [7, 8, 9], {"key1": "value1", "key2": "value2"}, "more text", 10, 11, 12, "last item")
    array_example = np.array([
        [1,2],
        [3,4],
        [5,6]
    ])
    # ==============write_data_to_excel_col/row-测试-start==============
    # write_data_to_excel_col(dic_example, save_file_path, 'dic_example')
    write_data_to_excel_col(list_example, save_file_path, 'list_example')
    write_data_to_excel_col(Tuple_example, save_file_path, 'tuple_example')
    write_data_to_excel_col(array_example, save_file_path, 'array_example')

    write_data_to_excel_row(dic_example, save_file_path)
    # write_data_to_excel_row(list_example, save_file_path)
    # write_data_to_excel_row(Tuple_example, save_file_path)
    write_data_to_excel_row(array_example, save_file_path)
    # ==============write_data_to_excel-测试-end================