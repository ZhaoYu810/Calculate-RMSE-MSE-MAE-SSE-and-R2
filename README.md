脚本可以读取excel文件内的想要的两列，并对这两列计算误差和相关性。

需要修改的参数：
    input_path = "文件的具体路径“
    sheet_name = "Sheet1" 
    column_name_ori = "需要比对的列A"
    column_name_deal = "需要比对的列B"
    list_length = 长度（读取进来的列A和列B都是个列表，这可以决定列表的长度）
    
需要运行时运行“calculate_err.py”
=========================================
Script can read the desired two columns in the Excel file and calculate the error and correlation for these two columns.

Parameters to be modified:

    input_path = "The specific path of the file"
    sheet_name = "Sheet1" 
    column_name_ori = "Column A to be compared"
    column_name_deal = "Column B to be compared"
    list_length = Length (Both Column A and Column B read in are lists, and this can determine the length of the lists)

Run "calculate_err.py" when needed. 
