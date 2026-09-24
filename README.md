# 小样本电商复购预测与解释分析演示系统

这是用于硕士论文开题答辩的 Streamlit 演示系统。页面包括复购概览、数据与模型结果、单笔预测交互和 SHAP 解释图。

## 在线部署

本仓库以 `app.py` 为入口，`requirements.txt` 为 Python 依赖声明，可直接在 Streamlit Community Cloud 选择仓库的 `main` 分支和 `app.py` 部署。建议使用 Python 3.12。

## 本地运行

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

演示账号和密码显示在登录页。

## 结果口径

表中的 TabPFN 指标与 SHAP 图片是既有实验结果的展示。此轻量在线版本未随仓库分发 TabPFN 权重，交互式单笔概率由规则化估计生成；两者不应混为同一次模型推断。展示数据来自项目中用于研究的去标识化数据副本。
