# Media Bias Detector

**Media Bias Detector** is a tool designed to analyze text for potential **bias**. It uses a **Logistic Regression model** that has been trained on a **custom dataset** of biased and neutral text. The tool classifies input text as either **biased** or **neutral** and highlights the words that contribute to the model’s prediction.

### **How It Works:**
1. **Input:**
   - Paste any **sentence or paragraph** into the tool for analysis.
   - Paste a **URL** to analyze the content of an entire article.

2. **Text Preprocessing:**
   - The input text is **cleaned** by converting it to lowercase and removing punctuation.
   - A **TF-IDF vectorizer** is used to convert the text into a numerical format.

3. **Bias Detection:**
   - A **Logistic Regression model** classifies the text as **biased** or **neutral** based on the numerical representation of the text.
   - The model provides a **confidence score** indicating the likelihood of the text being biased.

4. **Word Highlighting:**
   - Words contributing to bias are highlighted in **red**, while neutralizing words are highlighted in **green**.
   - These words are evaluated based on their **TF-IDF weights**.

5. **User Feedback:**
   - Users can provide feedback on the model’s predictions (correct, incorrect, or unsure), which is **logged** for future model improvements.

### **Dataset:**
The tool uses the **Media Bias Including Characteristics (MBIC)** dataset, which is a **publicly available dataset** containing articles labeled as **biased** or **neutral**. The dataset helps train the model to detect bias in media content.

- **Dataset Source:** [Zenodo - MBIC Dataset](https://zenodo.org/records/10547907)
- **Labels:** Biased (1), Neutral (0)

### **Model:**
The tool uses a **Logistic Regression model** that has been trained using the **MBIC dataset**. The model is designed to predict whether the input text contains bias.

- **Model Type:** Logistic Regression
- **Vectorizer:** **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorizer for transforming text into a numerical representation.

### **Tools Used:**
- **Scikit-learn:** For building the **Logistic Regression model** and **TF-IDF vectorizer**.
- **Pandas:** For handling data, logging feedback, and working with CSV files.
- **Streamlit:** For creating the interactive user interface to analyze text and feedback.
- **Newspaper3k:** For extracting text from news articles when provided with a URL.
- **Joblib:** For saving and loading the trained model and vectorizer.

### **Key Features:**
- **Custom Text Analysis:** Paste a sentence or paragraph to check for bias.
- **URL Article Analysis:** Paste a URL to analyze the content of an article for bias.
- **Bias Classification:** Predicts whether the text is **biased** or **neutral** with a confidence score.
- **Word Highlighting:** Highlights biased or neutralizing words in the text.
- **User Feedback:** Collects feedback from users to improve the model over time.

### **Installation and Usage:**
To run this project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/media-bias-detector.git


<img width="1133" alt="Screen Shot 2025-04-09 at 5 22 56 PM" src="https://github.com/user-attachments/assets/440cb2d9-8ebd-4db6-8217-4ad8d262ca3e" />
<img width="960" alt="Screen Shot 2025-04-09 at 5 23 01 PM" src="https://github.com/user-attachments/assets/0915282b-c770-4365-954c-85561cde4cd7" />
<img width="838" alt="Screen Shot 2025-04-09 at 5 23 19 PM" src="https://github.com/user-attachments/assets/e5117c9e-98d5-40a1-8f94-734bf32d60d6" />
<img width="783" alt="Screen Shot 2025-04-09 at 5 23 24 PM" src="https://github.com/user-attachments/assets/021ae68c-bba4-4230-8aa0-276c9daadeef" />
<img width="849" alt="Screen Shot 2025-04-09 at 5 25 23 PM" src="https://github.com/user-attachments/assets/0c8be68f-9f23-43e3-9ad2-98456f65bbd7" />
