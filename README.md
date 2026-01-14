# OLS Model Builder GUI

A python-based GUI which simplifies the process of specifying an regression model which can be used to predict missing values in a dataset. This includes an step-by-step model formula specification process, automated outlier detection and thresholding, and producing model summary tables and diagnostic plots. 

Note because this has automatic outlier removal, the assumption is that the missing values which are imputed are not outliers. 
For example, fitting missing values from cell suppression on skewed data. The outliers represent large cells, but the missing values are likely to be from small valued cells. The model descibes small value cells better by removing the large outliers.

## Overview

This tool provides a user-friendly interface for:
- Loading CSV data
- Building regression models interactively or via formula syntax
- Applying variable transformations (log, sqrt, polynomial, categorical)
- Detecting and removing outliers using Cook's Distance and Studentized Residuals
- Generating diagnostic plots
- Predicting missing response values
- Saving results to CSV

## Features

### Data Loading
- Import CSV files with automatic column detection
- Select response variable from available columns
- Automatic filtering to variables without missing values when response is missing

### Model Building Options

#### 1. Step-by-Step Builder (Recommended for beginners)
- Visual interface for selecting variables
- Checkbox selection for predictors
- Dropdown menus for transformations:
  - None (use variable as-is)
  - Categorical (treat as factor)
  - Log transformation
  - Square root transformation
  - Polynomial (specify degree up to 3)
- Interactive selection of two-way interaction terms
- Response variable transformation options

#### 2. Direct Formula Entry (For advanced users)
- Use R-style formula syntax
- Supports statsmodels formula notation (see https://www.statsmodels.org/stable/example_formulas.html)
- Examples:
  ```
  y ~ x1 + x2 + x3
  y ~ x1 + C(x2) + np.log(x3)
  y ~ x1 + x2 + x1:x2
  np.log(y) ~ x1 + poly(x2, 2)
  ```
### Outlier Detection & Removal
- **Cook's Distance Threshold**: Identifies influential observations (default: 1.0)
- **Studentized Residual Threshold**: Detects extreme residuals (default: 3.0)
- Automatic model refitting after outlier removal
- Summary statistics showing number of outliers detected

### Diagnostic Plots
- **Q-Q Plot**: Assesses normality of residuals
- **Residual vs. Fitted Plot**: Checks for heteroscedasticity and patterns
- Optional save to file location
- Embedded display in GUI

### Missing Value Prediction
- Predict missing response values using fitted model
- Generates prediction standard errors
- Saves complete dataset with predictions
- Preview window for results

## Installation

### Required Dependencies

```bash
pip install pandas numpy scipy matplotlib statsmodels formulaic
```
Optionally, seaborn can be used as well. The GUI will work without seaborn, but diagnostic plots will be simpler.

## Usage

### Quick Start

1. **Run the application**:
   ```bash
   python fixed_gui.py
   ```

2. **Load your data**:
   - Click "Load CSV"
   - Select your CSV file
   - Enter the name of your response variable when prompted

3. **Build your model**:
   - **Option A**: Click "Open step-by-step model formula builder"
     - Check variables to include
     - Select transformations from dropdowns
     - Add interaction terms if needed
     - Click "Build formula"
   
   - **Option B**: Type formula directly in the text box and click "Use direct model formula"

4. **Configure diagnostics** (optional):
   - Adjust Cook's Distance threshold (default: 1.0)
   - Adjust Studentized Residual threshold (default: 3.0)
   - Choose folder to save diagnostic plots (optional)

5. **Fit the model**:
   - Click "Fit model now"
   - Wait for model to fit (progress bar will animate)
   - Review model summary and diagnostic plots

6. **Make predictions**:
   - Set random seed for reproducibility
   - Click "Proceed to custom predictions"
   - Choose save location for output CSV
   - Review prediction preview

### Model Fitting Details

**Adjusting Outlier Thresholds**:

- **Cook's Distance**: 
  - Values > 1.0 are typically influential
  - Lower threshold (0.5) for stricter outlier removal
  - Higher threshold (1.5) for more lenient removal

- **Studentized Residuals**:
  - Values > 3.0 are typically extreme
  - Use 2.5 for stricter removal
  - Use 3.5 for more lenient removal

**Outlier Handling**
- Always check why observations are outliers
- Don't automatically remove all flagged points (set extremely high thresholds to not remove any outliers)
- Consider if outliers represent real phenomena
- Document outlier removal decisions


**Missing Value Handling**

- **Predictor variables**: Rows with missing predictors are removed
- **Response variable**: Missing values can be predicted after model fitting
- **Categorical variables**: Missing values converted to string "nan" category


### Model Interpretation

**Model Summary** includes:
- Coefficient estimates
- Standard errors
- t-statistics and p-values
- R-squared and Adjusted R-squared
- F-statistic
- Number of observations used

**Diagnostic Plots**:

1. **Q-Q Plot** (left):
   - Points should roughly follow the diagonal line
   - Systematic deviation indicates non-normal residuals
   - S-shaped curve suggests skewness
   - U-shaped curve suggests heavy tails

2. **Residual vs. Fitted** (right):
   - Points should scatter randomly around horizontal line at 0
   - Funnel shape indicates heteroscedasticity
   - Curved pattern suggests non-linear relationship
   - Red lowess line helps visualize patterns

**Model Validation**
- Check R-squared (but don't overfit!)
- Examine residual plots carefully
- Verify coefficient signs make sense
- Test predictions on holdout data if possible


## Limitations

- Only supports OLS (linear regression)
- No built-in cross-validation
- No automatic variable selection
- Limited to two-way interactions
- Assumes linear relationships (or transformable to linear)
- Requires complete predictors when response is missing

## References

- [Statsmodels Documentation](https://www.statsmodels.org/)
- [Statsmodels Formula Syntax](https://www.statsmodels.org/stable/example_formulas.html)
- [Formulaic Package](https://matthewwardrop.github.io/formulaic/)
- [Cook's Distance](https://online.stat.psu.edu/stat462/node/173/)
- [Studentized Residuals](https://online.stat.psu.edu/stat462/node/247/)


