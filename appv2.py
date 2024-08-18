import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import re
from io import BytesIO

st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Ranking-Based Conjoint Analyser by <a href='https://github.com/patilmanojkumar'>Manojkumar Patil</a></h1>",
    unsafe_allow_html=True
)
st.markdown(
    """
    <p align="center">
      <a href="https://github.com/DenverCoder1/readme-typing-svg">
        <img src="https://readme-typing-svg.herokuapp.com?font=Time+New+Roman&color=yellow&size=30&center=true&vCenter=true&width=600&height=100&lines=Conjoint+Analysis+Made+Simple!;rankconjoint_analyser-1.0;" alt="Typing SVG">
      </a>
    </p>
    """,
    unsafe_allow_html=True
)

# Upload data
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xlsx", "xls"])
if uploaded_file is not None:
    # Read the uploaded data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        # Allow user to select a sheet
        sheet_name = st.selectbox("Select sheet", pd.ExcelFile(uploaded_file).sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

    # Convert all column labels to lowercase and remove unnecessary characters/spaces
    df.columns = [re.sub(r'\W+', '_', col.lower().strip()) for col in df.columns]

    st.write("Data Preview:")
    st.write(df.head())

    # Let user select the ranking column
    ranking_column = st.selectbox("Select the ranking column", df.columns)
    
    # Let user select conjoint attributes
    conjoint_attributes = st.multiselect("Select attributes for conjoint analysis", [col for col in df.columns if col != ranking_column])

    # Conjoint attributes and model specification
    model = f'{ranking_column} ~ ' + ' + '.join([f'C({attr}, Sum)' for attr in conjoint_attributes])

    # Fit the model
    model_fit = smf.ols(model, data=df).fit()
    st.write(model_fit.summary())

    # Extracting part-worths and importance
    level_name = []
    part_worth = []
    part_worth_range = []
    important_levels = {}
    end = 1

    for item in conjoint_attributes:
        nlevels = len(list(np.unique(df[item])))
        level_name.append(list(np.unique(df[item])))

        begin = end
        end = begin + nlevels - 1

        new_part_worth = list(model_fit.params[begin:end])
        new_part_worth.append((-1) * sum(new_part_worth))
        important_levels[item] = np.argmax(new_part_worth)
        part_worth.append(new_part_worth)
        part_worth_range.append(max(new_part_worth) - min(new_part_worth))

    # Attribute importance
    attribute_importance = [round(100 * (i / sum(part_worth_range)), 2) for i in part_worth_range]

    # Creating a consolidated table
    combined_data = []
    for item, pw, levels, importance in zip(conjoint_attributes, part_worth, level_name, attribute_importance):
        for level, value in zip(levels, pw):
            combined_data.append([item, importance, level, value])

    combined_df = pd.DataFrame(combined_data, columns=['Attribute', 'Relative Importance', 'Level', 'Part Worth'])
    
    # Display the consolidated table
    st.write("Consolidated Part-Worth and Importance Table:")
    st.write(combined_df)
    
    # Allow user to download the consolidated table
    csv = combined_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Consolidated Table", data=csv, file_name='consolidated_part_worth_importance.csv', mime='text/csv')

    # Plotting relative importance of attributes
    plt.figure(figsize=(10, 5))
    sns.barplot(x=conjoint_attributes, y=attribute_importance)
    plt.title('Relative importance of attributes')
    plt.xlabel('Attributes')
    plt.ylabel('Importance')
    st.pyplot(plt)

   # Save plot to a BytesIO object in PNG format
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Allow user to download the plot
    st.download_button(label="Download Importance Plot", data=buf, file_name='importance_plot.png', mime='image/png')

    # Utility calculation
    part_worth_dict = {f"{item}_{level}": value for item, pw, levels in zip(conjoint_attributes, part_worth, level_name) for level, value in zip(levels, pw)}
    utility = [sum([part_worth_dict[f"{attr}_{df[attr][i]}"] for attr in conjoint_attributes]) for i in range(df.shape[0])]
    df['utility'] = utility

    # Profile with the highest utility score
    best_profile = df.iloc[np.argmax(utility)]
    st.write("The profile that has the highest utility score:")
    st.write(best_profile)

    # Preferred levels in each attribute
    st.write("Preferred levels in each attribute:")
    for i, item in enumerate(conjoint_attributes):
        st.write(f"Preferred level in {item} is :: {level_name[i][important_levels[item]]}")

    # Allow user to download the dataset with utility scores
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Data with Utility Scores", data=csv, file_name='data_with_utility_scores.csv', mime='text/csv')
