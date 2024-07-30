# Invoice Document Similarity Matching System

## Overview

This project is designed to process and compare invoices to determine their similarity. By leveraging Optical Character Recognition (OCR) and text similarity metrics, the system extracts key information from invoices and uses it to identify similar documents. The main components of this system include PDF-to-image conversion, text extraction via OCR, and similarity computation based on extracted fields.

## Features

- **OCR-Based Text Extraction**: Converts invoice PDFs into images and extracts text using Tesseract OCR.
- **Field Extraction**: Identifies key details from invoices such as invoice number, date, total amount, and company name.
- **Text Similarity**: Calculates the similarity between invoices based on their textual representation.
- **Comparison Display**: Presents a tabular comparison of key fields and itemized details for easy review.
- **Accuracy Reporting**: Evaluates and reports the accuracy of the matching algorithm.
- **Result Export**: Saves similarity results to a CSV file for further analysis.

## Technologies Used

### Optical Character Recognition (OCR)

- **Tesseract OCR**: An open-source OCR engine used to extract text from images of invoices. It is particularly useful for handling various formats and languages.

### Document Processing

- **PyMuPDF**: A Python library used to convert PDF files into images. PyMuPDF (also known as Fitz) allows for efficient extraction of text from PDF pages.
- **Pillow (PIL)**: A Python Imaging Library used for image manipulation, including opening and processing images.

### Text Analysis

- **TfidfVectorizer**: Part of the Scikit-learn library, used to convert text into numerical features (TF-IDF vectors) for similarity calculation.
- **Cosine Similarity**: A metric used to determine the similarity between two text vectors based on the cosine of the angle between them.

### Data Handling

- **Pandas**: A library for data manipulation and analysis. Used to handle and save results in CSV format.
- **NumPy**: A library for numerical operations. Utilized for calculating accuracy metrics.

### Date Parsing

- **dateutil**: A library for parsing dates from text, which helps in interpreting different date formats found in invoices.

### Other Tools

- **Tabulate**: A library used to generate well-formatted tabular outputs for easy reading and comparison of results.

## Code Explanation

### PDF to Image Conversion

**Function**: `pdf_to_image(pdf_path)`

- Converts the first page of a PDF invoice into an image format.
- **Library Used**: PyMuPDF (fitz) for loading and converting PDF pages; Pillow for image handling.

### Invoice Details Extraction

**Function**: `extract_invoice_details(pdf_path)`

- Extracts key details from the invoice image using Tesseract OCR.
- **Field Extraction Patterns**:
  - **Invoice Number**: Uses regex patterns to locate various forms of invoice numbers.
  - **Date**: Identifies invoice dates in multiple formats and converts them to a standard format.
  - **Total Amount**: Finds the total amount due on the invoice.
  - **Company Name**: Extracts the company name from the text.
  - **Items**: Parses itemized details such as quantity, description, and prices.

### Similarity Calculation

**Function**: `calculate_similarity(details1, details2)`

- Computes the similarity between two invoices based on their extracted text using TF-IDF vectorization and cosine similarity.
- **Library Used**: Scikit-learn's `TfidfVectorizer` for text vectorization and `cosine_similarity` for similarity measurement.

### Invoice Processing

**Function**: `process_invoices(invoice_dir)`

- Processes all PDF invoices in a given directory to extract details and store them in a list.
- Handles errors gracefully and skips non-PDF files.

### Finding Similar Invoices

**Function**: `find_similar_invoices(test_invoice, database_invoices)`

- Compares a test invoice against a database of training invoices to find the most similar ones.
- Returns the top 5 most similar invoices based on similarity scores.

### Displaying Comparisons

**Function**: `display_invoice_comparison(test_invoice, similar_invoices)`

- Displays a tabular comparison of fields and itemized details between a test invoice and its most similar invoices.
- Uses the `tabulate` library for formatted output.

### Accuracy Reporting

**Function**: `generate_accuracy_report(test_invoices, train_invoices)`

- Calculates and reports the accuracy of the matching algorithm by comparing test invoices to training invoices.
- **Metrics**: Field-wise accuracy and overall accuracy.

### Result Export

**Function**: Saves similarity results to a CSV file for further review.
- **Library Used**: Pandas for data handling and CSV export.

## How to Run the Code

1. **Install Required Libraries**:
   ```bash
   pip install pytesseract pandas numpy scikit-learn python-dateutil PyMuPDF Pillow tabulate
   ```

2. **Set Up Tesseract**:
   - Download and install Tesseract OCR from [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract).
   - Adjust the `tesseract_cmd` path and `TESSDATA_PREFIX` environment variable in the code to match your installation paths.

3. **Prepare Data**:
   - Place your training invoice PDFs in the `train_dir` directory.
   - Place your test invoice PDFs in the `test_dir` directory.

4. **Run the Script**:
   - Execute the Python script. It will process the invoices, compute similarities, generate an accuracy report, display comparisons, and save results to `similarity_results.csv`.

5. **Review Results**:
   - Check `similarity_results.csv` for saved similarity results.
   - Review the printed comparison tables and accuracy reports in the console output.
     
## Example

Here's an example of how to use the script:

```python
train_dir = 'path/to/train_directory'
test_dir = 'path/to/test_directory'

# Process invoices
train_invoices = process_invoices(train_dir)
test_invoices = process_invoices(test_dir)

# Generate accuracy report
accuracy_report = generate_accuracy_report(test_invoices, train_invoices)
print(accuracy_report)

# Display comparisons
for test_invoice in test_invoices:
    similar_invoices = find_similar_invoices(test_invoice, train_invoices)
    display_invoice_comparison(test_invoice, similar_invoices)
```


## Accuracy and Efficiency

### Accuracy

The accuracy of the system is evaluated based on:
- **Field Accuracy**: Measures the correctness of specific fields (invoice number, date, total, company) between invoices.
- **Overall Accuracy**: The average accuracy across all test invoices.

### Efficiency

The system's efficiency is influenced by:
- **Processing Time**: Time taken for OCR, text extraction, and similarity calculation. Optimizations can include parallel processing or reducing image resolution for faster OCR.
- **Resource Usage**: The code uses moderate memory and CPU resources primarily for image processing and text extraction.

## Code Quality

The code is structured to be:
- **Clean**: With clear function names and organized logic.
- **Well-Commented**: Including explanations for each function and major code segments.
- **Modular**: Functions are designed to perform distinct tasks, facilitating maintenance and expansion.

---

# Technologies and Resources

## Technologies Used

1. **Tesseract OCR**
   - **Description**: Open-source OCR engine for text extraction from images.
   - **Website**: [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract)
   - **Installation**: Download from the GitHub page and follow installation instructions for your OS.

2. **PyMuPDF (fitz)**
   - **Description**: Library for working with PDF files, including converting PDF pages to images.
   - **Website**: [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
   - **Installation**: `pip install PyMuPDF`

3. **Pillow (PIL)**
   - **Description**: Python Imaging Library for image processing tasks.
   - **Website**: [Pillow Documentation](https://pillow.readthedocs.io/en/stable/)
   - **Installation**: `pip install Pillow`

4. **TfidfVectorizer (Scikit-learn)**
   - **Description**: Converts text into TF-IDF vectors for similarity measurement.
   - **Website**: [Scikit-learn Documentation](https://scikit-learn.org/stable/)
   - **Installation**: `pip install scikit-learn`

5. **Cosine Similarity (Scikit-learn)**
   - **Description**: Measures the similarity between two vectors.
   - **Website**: [Scikit-learn Documentation](https://scikit-learn.org/stable/)
   - **Installation**: Included with Scikit-learn.

6. **Pandas**
   - **Description**: Data manipulation and analysis library.
   - **Website**: [Pandas Documentation](https://pandas.pydata.org/docs/)
   - **Installation**: `pip install pandas`

7. **NumPy**
   - **Description**: Library for numerical operations and array handling.
   - **Website**: [NumPy Documentation](https://numpy.org/doc/stable/)
   - **Installation**: `pip install numpy`

8. **dateutil**
   - **Description**: Library for parsing dates in different formats.
   - **Website**: [dateutil Documentation](https://dateutil.readthedocs.io/)
   - **Installation**: `pip install python-dateutil`

9. **Tabulate**
   - **Description**: Generates formatted tabular outputs.
   - **Website**: [Tabulate Documentation](https://pypi.org/project/tabulate/)
   - **Installation**: `pip install tabulate`
 ### Future Improvements:
  **Enhanced OCR:**


### **Improvement Areas**

1. **OCR Accuracy:**
   - **Handling Varying Layouts:** Current OCR techniques may struggle with invoices that have unconventional layouts or formats. Future improvements could involve training custom OCR models specifically for diverse invoice templates. This could include adding more samples of different invoice types to the training dataset, enabling the model to better recognize and extract text from various formats.
   - **Poor-Quality Scans:** OCR performance can degrade with poor-quality scans or images. To address this, implementing image preprocessing techniques such as denoising, binarization, and contrast adjustment could enhance text clarity before OCR processing. Additionally, integrating advanced OCR models that are robust to noise and distortions could further improve accuracy.

2. **Machine Learning:**
   - **Advanced Techniques:** Incorporating machine learning models could greatly enhance the system’s capability to recognize and classify invoice details. Techniques such as Named Entity Recognition (NER) and deep learning-based text classification could be used to identify and extract key fields more accurately. For instance, training a neural network model to recognize specific invoice elements (like line items and amounts) can improve extraction accuracy. Additionally, models like BERT or GPT could be employed for better contextual understanding and entity extraction.
   - **Document Classification:** Machine learning could also be used to automatically classify invoices into categories (e.g., utilities, services, products), making it easier to process and compare invoices based on their type.

3. **User Interface:**
   - **UI Development:** Developing a user-friendly graphical user interface (GUI) would enhance the interaction experience. The interface could feature drag-and-drop functionality for uploading invoices, real-time feedback on the OCR extraction process, and intuitive visualization of comparison results. Implementing dashboards with summary statistics, error logs, and detailed views of extracted data would make the system more accessible and manageable for users.
   - **Interactive Features:** To further improve usability, consider adding interactive features such as customizable extraction templates, manual correction options, and detailed comparison reports. This would allow users to tailor the system to their specific needs and ensure high accuracy.

4. **Better Language Adaptability:**
   - **Multilingual Support:** To increase the system’s versatility, integrating support for multiple languages beyond German is essential. This involves extending the OCR system to handle various languages and scripts, and adapting regex patterns for different linguistic structures. Implementing multilingual OCR models and language-specific text processing pipelines can significantly broaden the system’s applicability.
   - **Localization:** Developing localization features that accommodate regional differences in invoice formats, date formats, and currency symbols will make the system more globally applicable. Customizing regex patterns and extraction rules for different locales can ensure accurate data extraction and comparison.

## Resources

- **Tesseract OCR Documentation**: [Tesseract Documentation](https://tesseract-ocr.github.io/tessdoc/)
- **PyMuPDF Documentation**: [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- **Pillow Documentation**: [Pillow Documentation](https://pillow.readthedocs.io/)
-

 **Scikit-learn Documentation**: [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- **Pandas Documentation**: [Pandas Documentation](https://pandas.pydata.org/docs/)
- **NumPy Documentation**: [NumPy Documentation](https://numpy.org/doc/stable/)
- **dateutil Documentation**: [dateutil Documentation](https://dateutil.readthedocs.io/)
- **Tabulate Documentation**: [Tabulate Documentation](https://pypi.org/project/tabulate/)


## Contact

For any questions or feedback, feel free to reach out to [devikaverma1554@gmail.com](mailto:devikaverma1554@gmail.com).


