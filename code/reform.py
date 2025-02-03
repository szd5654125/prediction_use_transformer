import nbformat

def convert_ipynb_to_py(ipynb_file, py_file):
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    with open(py_file, 'w', encoding='utf-8') as f:
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                f.write('# %%\n')  # 用于区分代码块
                f.write(cell.source + '\n\n')

# 示例
convert_ipynb_to_py('bitcoin_price_prediction_optuna.ipynb', 'bitcoin_price_prediction_optuna.py')