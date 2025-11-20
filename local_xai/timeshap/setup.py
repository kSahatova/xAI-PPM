from setuptools import setup, find_packages

setup(
    name="timeshap",
    version="0.1.0",
    packages=[
        "timeshap.explainer",
        "timeshap.wrappers",
        "timeshap.utils",
        "timeshap.plot",
    ],
    python_requires=">=3.11",
    author="Bento, Jo\~{a}o and Saleiro, Pedro and Cruz, Andr'{e} F. and Figueiredo, M'{a}rio A.T. and Bizarro, Pedro",
    description="Calculation of SHAP value explanations at the subsequence level",
)
