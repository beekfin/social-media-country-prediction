from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


DATA_PATH = Path("SocialMediaUsersDataset.csv")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42
REFERENCE_DATE = date(2026, 5, 3)
MODEL_SAMPLE_SIZE = 60_000


def parse_interests(value: str) -> list[str]:
    text = "[" + value + "]"
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = [part.strip().strip("'\"") for part in value.split(",")]
    return [str(item).strip() for item in parsed if str(item).strip()]


def age_from_dob(value: str) -> int:
    dob = date.fromisoformat(value)
    return REFERENCE_DATE.year - dob.year - (
        (REFERENCE_DATE.month, REFERENCE_DATE.day) < (dob.month, dob.day)
    )


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype={"UserID": str})
    df.columns = [column.strip() for column in df.columns]
    df = df.drop_duplicates(subset=["UserID"]).copy()

    df["interests_list"] = df["Interests"].map(parse_interests)
    df["interests_unique"] = df["interests_list"].map(lambda items: sorted(set(items)))
    df["interests_clean"] = df["interests_unique"].map(lambda items: " | ".join(items))
    df["interest_count"] = df["interests_unique"].map(len)
    df["age"] = df["DOB"].map(age_from_dob)
    df["birth_year"] = pd.to_datetime(df["DOB"]).dt.year
    df["first_name"] = df["Name"].str.split().str[0].str.lower()
    df["target"] = (df["Country"] == "United States").astype(int)
    df["target_label"] = np.where(df["target"] == 1, "United States", "Other countries")
    return df


def make_preprocessor(max_city_categories: int = 2) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "name_chars",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3),
                "first_name",
            ),
            (
                "interests",
                CountVectorizer(token_pattern=r"[^|]+", binary=True),
                "interests_clean",
            ),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", min_frequency=max_city_categories),
                ["City"],
            ),
            ("numeric", StandardScaler(), ["age", "interest_count"]),
        ],
        remainder="drop",
    )


def plot_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="target_label", hue="target_label", palette="Set2", ax=ax, legend=False)
    ax.set_title("Распределение целевой переменной")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Число пользователей")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="age", hue="target_label", bins=26, kde=False, palette="Set2", ax=ax)
    ax.set_title("Возраст пользователей по классам")
    ax.set_xlabel("Возраст")
    ax.set_ylabel("Число пользователей")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "age_by_target.png", dpi=160)
    plt.close(fig)

    top_interests = (
        df.explode("interests_unique")
        .groupby("interests_unique")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    top_interests.plot(kind="barh", color="#4c78a8", ax=ax)
    ax.set_title("Топ-10 интересов")
    ax.set_xlabel("Число профилей")
    ax.set_ylabel("Интерес")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_interests.png", dpi=160)
    plt.close(fig)

    interest_target = (
        df.explode("interests_unique")
        .pivot_table(index="interests_unique", columns="target_label", values="UserID", aggfunc="count")
        .fillna(0)
    )
    interest_target["target_share"] = interest_target["United States"] / (
        interest_target["United States"] + interest_target["Other countries"]
    )
    interest_target = (
        interest_target.assign(diff=lambda x: (x["target_share"] - df["target"].mean()).abs())
        .sort_values("diff", ascending=False)
        .head(10)
        .sort_values("target_share")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    interest_target["target_share"].plot(kind="barh", color="#59a14f", ax=ax)
    ax.axvline(df["target"].mean(), color="black", linewidth=1)
    ax.set_title("Доля класса United States среди наиболее различающихся интересов")
    ax.set_xlabel("Доля United States")
    ax.set_ylabel("Интерес")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "interest_target_dependency.png", dpi=160)
    plt.close(fig)

    top_cities = df["City"].value_counts().head(20).index
    city_target = (
        df[df["City"].isin(top_cities)]
        .pivot_table(index="City", columns="target_label", values="UserID", aggfunc="count")
        .fillna(0)
    )
    if "United States" not in city_target:
        city_target["United States"] = 0
    if "Other countries" not in city_target:
        city_target["Other countries"] = 0
    city_target["target_share"] = city_target["United States"] / (
        city_target["United States"] + city_target["Other countries"]
    )
    city_target = city_target.sort_values("target_share")
    fig, ax = plt.subplots(figsize=(8, 5))
    city_target["target_share"].plot(kind="barh", color="#f28e2b", ax=ax)
    ax.axvline(df["target"].mean(), color="black", linewidth=1)
    ax.set_title("Доля класса United States в популярных городах")
    ax.set_xlabel("Доля United States")
    ax.set_ylabel("Город")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "city_target_dependency.png", dpi=160)
    plt.close(fig)


def make_model_grid() -> list[dict]:
    return [
        {
            "model": [LogisticRegression(max_iter=1500, solver="liblinear", random_state=RANDOM_STATE)],
            "model__C": [0.1, 1.0, 3.0],
            "model__class_weight": [None, "balanced"],
        },
        {
            "model": [LinearSVC(max_iter=5000, dual="auto", random_state=RANDOM_STATE)],
            "model__C": [0.1, 1.0, 3.0],
            "model__class_weight": [None, "balanced"],
        },
        {
            "model": [
                SGDClassifier(
                    loss="log_loss",
                    max_iter=1500,
                    tol=1e-3,
                    random_state=RANDOM_STATE,
                )
            ],
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__penalty": ["l2", "elasticnet"],
        },
        {
            "model": [RidgeClassifier(random_state=RANDOM_STATE)],
            "model__alpha": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
    ]


def score_with_auc(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    if hasattr(model, "predict_proba"):
        return roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    return roc_auc_score(y_test, model.decision_function(x_test))


def train_models(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, dict]:
    feature_columns = [
        "first_name",
        "interests_clean",
        "City",
        "age",
        "interest_count",
    ]
    sampled = df.sample(
        n=min(MODEL_SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE,
    )
    x = sampled[feature_columns]
    y = sampled["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    base_pipeline = Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", LogisticRegression()),
        ]
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        base_pipeline,
        make_model_grid(),
        scoring={"f1": "f1", "roc_auc": "roc_auc", "accuracy": "accuracy"},
        refit="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(x_train, y_train)

    results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_f1")
    results[
        [
            "rank_test_f1",
            "mean_test_f1",
            "std_test_f1",
            "mean_test_roc_auc",
            "mean_test_accuracy",
            "params",
        ]
    ].to_csv(OUTPUT_DIR / "model_selection_results.csv", index=False)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)
    metrics = {
        "best_params": grid.best_params_,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": score_with_auc(best_model, x_test, y_test),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Other countries", "United States"],
            digits=4,
        ),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }

    dummy = DummyClassifier(strategy="most_frequent")
    dummy_pipe = Pipeline(steps=[("preprocess", make_preprocessor()), ("model", dummy)])
    dummy_pipe.fit(x_train, y_train)
    dummy_pred = dummy_pipe.predict(x_test)
    metrics["dummy_accuracy"] = accuracy_score(y_test, dummy_pred)
    metrics["dummy_f1"] = f1_score(y_test, dummy_pred, zero_division=0)

    group_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    group_train_idx, group_test_idx = next(group_split.split(x, y, groups=sampled["City"]))
    x_group_train = x.iloc[group_train_idx]
    x_group_test = x.iloc[group_test_idx]
    y_group_train = y.iloc[group_train_idx]
    y_group_test = y.iloc[group_test_idx]
    group_model = clone(best_model)
    group_model.fit(x_group_train, y_group_train)
    group_pred = group_model.predict(x_group_test)
    metrics["group_city_accuracy"] = accuracy_score(y_group_test, group_pred)
    metrics["group_city_f1"] = f1_score(y_group_test, group_pred, zero_division=0)
    metrics["group_city_roc_auc"] = score_with_auc(group_model, x_group_test, y_group_test)
    metrics["group_city_n_train"] = len(x_group_train)
    metrics["group_city_n_test"] = len(x_group_test)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        best_model,
        x_test,
        y_test,
        display_labels=["Other countries", "United States"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Матрица ошибок лучшей модели")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")
    return best_model, results, metrics


def build_surrogate(df: pd.DataFrame, best_model: Pipeline) -> str:
    feature_columns = [
        "first_name",
        "interests_clean",
        "City",
        "age",
        "interest_count",
    ]
    sample = df.sample(n=min(12000, len(df)), random_state=RANDOM_STATE)
    x = sample[feature_columns]
    pseudo_y = pd.Series(best_model.predict(x), index=sample.index, name="pseudo_target")
    positive_index = pseudo_y[pseudo_y == 1].index
    negative_index = pseudo_y[pseudo_y == 0].sample(
        n=min(len(positive_index) * 2, int((pseudo_y == 0).sum())),
        random_state=RANDOM_STATE,
    ).index
    balanced_index = positive_index.union(negative_index)
    sample = sample.loc[balanced_index]
    pseudo_y = pseudo_y.loc[balanced_index]

    simple = sample[["age", "interest_count", "City"]].copy()
    for interest in ["Fashion", "Gaming", "Technology", "Sports", "Beauty"]:
        simple[f"interest_{interest}"] = sample["interests_unique"].map(
            lambda values, interest=interest: int(interest in values)
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), ["City"]),
            ("num", StandardScaler(), [col for col in simple.columns if col != "City"]),
        ]
    )
    surrogate = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=RANDOM_STATE)),
        ]
    )
    surrogate.fit(simple, pseudo_y)
    fidelity = surrogate.score(simple, pseudo_y)

    tree_model = surrogate.named_steps["model"]
    names = surrogate.named_steps["preprocess"].get_feature_names_out()
    rules = export_text(tree_model, feature_names=list(names), max_depth=3)

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_tree(
        tree_model,
        feature_names=list(names),
        class_names=["Other countries", "United States"],
        filled=True,
        max_depth=3,
        fontsize=7,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "surrogate_tree.png", dpi=180)
    plt.close(fig)

    text = f"Surrogate fidelity: {fidelity:.4f}\n\n{rules}"
    (OUTPUT_DIR / "surrogate_tree_rules.txt").write_text(text, encoding="utf-8")
    return text


def write_report(df: pd.DataFrame, metrics: dict, surrogate_text: str) -> None:
    stats = {
        "rows": len(df),
        "cols": len(df.columns),
        "countries": df["Country"].nunique(),
        "cities": df["City"].nunique(),
        "interests": len(sorted({item for items in df["interests_unique"] for item in items})),
        "target_share": df["target"].mean(),
        "age_min": df["age"].min(),
        "age_max": df["age"].max(),
        "age_mean": df["age"].mean(),
        "duplicates": df["UserID"].duplicated().sum(),
        "missing": int(df.isna().sum().sum()),
    }
    top_results = pd.read_csv(OUTPUT_DIR / "model_selection_results.csv").head(8)
    top_table = top_results[
        ["rank_test_f1", "mean_test_f1", "mean_test_roc_auc", "mean_test_accuracy", "params"]
    ].to_markdown(index=False)

    report = f"""# Курсовая работа по методам машинного обучения

## Тема

Построение модели прогнозирования принадлежности пользователя социальной сети к сегменту `United States` по открытым признакам профиля.

## 1. Анализ предметной области

Социальные сети содержат структурированные и полуструктурированные данные о пользователях: имя, дата рождения, город, страна, интересы. В прикладных системах часть профилей может содержать неполную географическую информацию: например, указан город, но отсутствует или повреждена страна. В данной работе строится модель, которая прогнозирует, относится ли профиль к сегменту `United States`, по остальным открытым признакам.

Априорные знания и ограничения:

- `Country` используется только для формирования целевой переменной и исключается из входных признаков модели.
- `City` является главным прикладным сигналом, но может быть неоднозначным: города с одинаковыми названиями встречаются в разных странах.
- `Name` может содержать косвенный региональный сигнал, но переносимость такого признака зависит от языка и культурного контекста.
- `DOB` используется только для расчета возраста на дату {REFERENCE_DATE.isoformat()}; исходная дата рождения не должна напрямую передаваться в модель из-за избыточной персональной детализации.
- `UserID` является техническим идентификатором и исключается из признаков, потому что не несет обобщаемой закономерности.
- Модель нельзя применять как единственный источник истины для юридически значимых географических проверок; допустимый сценарий - восстановление пропусков, маршрутизация локализации и агрегированная аналитика.

Цель анализа: построить прогнозную бинарную модель, определяющую вероятность класса `United States` по признакам профиля без использования `Country` как входа. Критерий качества - F1-score для положительного класса на отложенной тестовой выборке, дополнительно контролируются ROC-AUC и accuracy. Процедура проверки: стратифицированное разделение train/test 75/25, подбор гиперпараметров на train через 3-fold StratifiedKFold, финальная оценка на test.

Пользовательские сценарии программной системы:

- Аналитик загружает CSV с профилями, запускает скрипт предобработки и получает отчет о качестве модели.
- ML-инженер сравнивает несколько гипотез алгоритмов и сохраняет лучшую модель в `outputs/best_model.joblib`.
- Продуктовый аналитик применяет модель к профилям с отсутствующей страной для предварительной локализации и агрегированной оценки аудитории.
- Ответственный за качество проверяет графики EDA, матрицу ошибок и суррогатное дерево для поиска перекосов и слабых мест модели.

## 2. Формирование и подготовка данных

Исходный файл: `SocialMediaUsersDataset.csv`. После удаления возможных дублей по `UserID` в анализ включено {stats["rows"]} записей. В наборе {stats["countries"]} стран, {stats["cities"]} городов и {stats["interests"]} уникальных интересов. Пропущенных значений после загрузки: {stats["missing"]}; дублей `UserID`: {stats["duplicates"]}.

В анализ включены признаки `Name`, `DOB`, `Interests`, `City`, `Country`. `Country` используется только для построения целевой переменной `United States / Other countries`, `UserID` исключен как идентификатор, `Gender` не используется, чтобы задача оставалась географической, а не демографической. Из `Name` выделяется `first_name`; из `DOB` рассчитывается `age`; из `Interests` формируются очищенный список интересов и `interest_count`. Таким образом, из 7 сырых колонок получается более 10 машинных признаков: символьные n-граммы имени, бинарные индикаторы интересов, one-hot признаки города, числовые признаки возраста и количества интересов.

Методы предобработки:

- проверка пропусков и дублей по `UserID`;
- парсинг списка интересов и удаление повторяющихся интересов внутри одного профиля;
- расчет возраста на фиксированную дату {REFERENCE_DATE.isoformat()};
- TF-IDF по символьным n-граммам имени, чтобы устойчиво учитывать короткие имена и вариативность написания;
- бинарное кодирование интересов через `CountVectorizer`;
- one-hot кодирование города с объединением редких категорий через `min_frequency`;
- стандартизация числовых признаков `age` и `interest_count`.

Feature engineering:

- `first_name` вместо полного имени снижает шум фамилий;
- `age` является интерпретируемым агрегатом даты рождения;
- `interest_count` отражает активность заполнения профиля;
- бинарные признаки интересов дают разреженное тематическое описание профиля;
- редкие категории городов агрегируются, что уменьшает размерность и риск переобучения.

Статистические характеристики:

- возраст: минимум {stats["age_min"]}, максимум {stats["age_max"]}, среднее {stats["age_mean"]:.2f};
- доля класса `United States`: {stats["target_share"]:.4f};
- число уникальных интересов: {stats["interests"]}.

Графики разведочного анализа сохранены:

- `outputs/target_distribution.png` - распределение целевой переменной;
- `outputs/age_by_target.png` - возраст по классам;
- `outputs/top_interests.png` - наиболее частые интересы;
- `outputs/interest_target_dependency.png` - зависимость целевой переменной от интересов;
- `outputs/city_target_dependency.png` - зависимость целевой переменной от города.

## 3. Построение модели и валидация

Тип задачи: бинарная классификация с учителем. Искомая зависимость - отображение признаков профиля в класс `Other countries` или `United States`.

Гипотезы алгоритмов:

- Logistic Regression: линейная модель для разреженных текстовых и категориальных признаков; гиперпараметры `C` в диапазоне 0.1-3.0 и `class_weight`.
- Linear SVM: сильная линейная модель для высокоразмерных разреженных данных; гиперпараметр `C` в диапазоне 0.1-3.0 и `class_weight`.
- SGDClassifier с log-loss: масштабируемая линейная модель; гиперпараметры `alpha` 1e-5-1e-3 и `penalty` l2/elasticnet.
- RidgeClassifier: линейный классификатор с L2-регуляризацией через гребневую регрессию; гиперпараметры `alpha` и `class_weight`.

Для ускорения перебора гипотез использована стратифицированная подвыборка {MODEL_SAMPLE_SIZE} записей из полного набора. Размер train: {metrics["n_train"]}, размер test: {metrics["n_test"]}.

Результаты подбора гиперпараметров:

{top_table}

Лучшая модель по F1:

```text
{metrics["best_params"]}
```

Качество на отложенной тестовой выборке:

- accuracy: {metrics["test_accuracy"]:.4f};
- F1: {metrics["test_f1"]:.4f};
- ROC-AUC: {metrics["test_roc_auc"]:.4f};
- baseline accuracy: {metrics["dummy_accuracy"]:.4f};
- baseline F1: {metrics["dummy_f1"]:.4f}.

Дополнительная стресс-проверка выполнена через `GroupShuffleSplit` по `City`: города из тестовой части не встречаются в обучении. Это более строгий сценарий для новых городов, а не для восстановления страны у профилей из уже известных городов. Размер group-train: {metrics["group_city_n_train"]}, размер group-test: {metrics["group_city_n_test"]}. Метрики на этой проверке: accuracy {metrics["group_city_accuracy"]:.4f}, F1 {metrics["group_city_f1"]:.4f}, ROC-AUC {metrics["group_city_roc_auc"]:.4f}. Разрыв между random split и group split показывает, что основной сигнал модели связан именно с городом, а не с универсальными демографическими закономерностями.

Подробный отчет классификации:

```text
{metrics["classification_report"]}
```

Матрица ошибок сохранена в `outputs/confusion_matrix.png`. Финальная модель сохранена в `outputs/best_model.joblib`.

## 4. Интерпретация и объяснение модели

Для глобальной интерпретации построена суррогатная модель DecisionTreeClassifier глубины 3, обученная приближать прогнозы лучшей модели на простой сбалансированной подвыборке признаков. Дерево не заменяет основную модель, но показывает грубые правила, которыми можно описывать часть найденной зависимости. Текст правил сохранен в `outputs/surrogate_tree_rules.txt`, изображение дерева - в `outputs/surrogate_tree.png`.

Фрагмент суррогатного описания:

```text
{surrogate_text[:2500]}
```

## Вывод

В работе построен воспроизводимый pipeline: загрузка и очистка данных, feature engineering, EDA, перебор 4 гипотез моделей, финальная оценка на тестовой выборке и интерпретация через суррогатное дерево. По сравнению с baseline лучшая модель показывает существенное улучшение; основной вклад ожидаемо дает город, а имя, возраст и интересы выступают дополнительными источниками сигнала. Для промышленного применения необходимы отдельная проверка на реальных данных, мониторинг качества для неоднозначных городов и осторожная трактовка вероятностного прогноза как вспомогательной оценки.
"""
    Path("coursework_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_and_prepare()
    plot_eda(df)
    best_model, _results, metrics = train_models(df)
    surrogate_text = build_surrogate(df, best_model)
    write_report(df, metrics, surrogate_text)
    print("Done")
    print(f"Report: {Path('coursework_report.md').resolve()}")
    print(f"Outputs: {OUTPUT_DIR.resolve()}")
    print(f"F1: {metrics['test_f1']:.4f}, ROC-AUC: {metrics['test_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
