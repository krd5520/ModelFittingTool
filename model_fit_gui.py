import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import threading
import webbrowser
from formulaic import Formula
import statsmodels.api as sm
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re

## Need seaborn and statsmodels.graphics.gofplots if plotting
matplotlib.use('TkAgg')

# Define TRANSFORMS constant
TRANSFORMS = ['none', 'categorical', 'log', 'sqrt']


# sys.path.append(os.path.abspath("./functions/"))
# from model_build import possible_variables
## Find variables that can be used to impute missing response entries.
# i.e. variables that are never missing when the response is and that take more than 1 unique value.
def possible_variables(data, response):
    naidx = data[response].isna()  # missing response
    missing_y = data[naidx]  # subset of data where response is missing
    possible_vars = list()  # initialize
    for cvar in data.columns:  # for each column
        if cvar == response:  # skip response variable
            pass
        else:
            if missing_y[cvar].isna().sum() == 0:  # none missing when response is missing
                if data[cvar].nunique() > 1:  # more than 1 unique value
                    possible_vars.append(cvar)  # add to list of possible model variables
    return possible_vars


def model_subdata(data, formula_str, remove_missing_pred=True):
    # Retrieve OLS formula from config.yaml
    formula = Formula(formula_str)
    checkvars = formula.required_variables
    assert checkvars.issubset(
        set(data.columns)), f"Formula variables must be columns of the data. {checkvars - set(data.columns)} are not columns in data."
    data = data.loc[:, list(checkvars)]  # keep only model variables
    has_na = data.columns[data.isna().any()].to_list()
    if has_na is not None and len(has_na) > 0:
        if not remove_missing_pred:  # remove any row with an NA value
            yname = [y for y in list(checkvars) if y in formula.lhs]
            assert len(yname) == 1, f"response variable must be on left side of '~' in the formula: {formula_str}"
            prednames = [x for x in list(checkvars) if x not in yname]
            cat_predvars = find_categorical(data, formula_str, prednames)
            data.loc[:, cat_predvars] = data.loc[:, cat_predvars].astype("str")
            has_na_not_cat = [x for x in has_na if x not in cat_predvars + yname]
            if has_na_not_cat is not None and len(has_na_not_cat) > 0:
                warnings.warn(
                    "Some numeric predictors have missing values that cannot be turned into categorical values:" + ", ".join(
                        has_na_not_cat))
                data.dropna(axis=0, how='any', subset=yname + has_na_not_cat)
    data.dropna(axis=0, how='any', inplace=True)
    return data


def find_categorical(data, formula_str, predvars):
    dtadtype = data.dtypes
    cat_vars = dtadtype.index[dtadtype == "object"].to_list()
    formula_str = formula_str.replace(" ", "")
    force_cat_vars = [varname for varname in predvars if "C(" + varname in formula_str]
    cat_vars = cat_vars + force_cat_vars
    return cat_vars


def get_model(data, formula, cooks_threshold, studentr_threshold, diagnostic_path=None,
              remove_missing_pred=True, return_diagnostics=True):
    """
    Fit OLS model with outlier detection and removal.

    Returns:
        model: fitted statsmodels OLS model
        outlier_text: list of strings describing outliers
        summary_str: model summary as string
        fitted_values: fitted values from model
        residuals: residuals from model
    """
    subdata = model_subdata(data, formula_str=formula, remove_missing_pred=remove_missing_pred)

    # Create design matrices and perform initial model fitting
    y, X = Formula(formula).get_model_matrix(subdata)
    model = sm.OLS(y, X).fit()

    # Outlier detection
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    studentized_resid = influence.resid_studentized_internal

    # Identify outliers
    outlier_mask_cooks = cooks_d > cooks_threshold
    outlier_mask_resid = np.abs(studentized_resid) > studentr_threshold
    outlier_mask = outlier_mask_cooks | outlier_mask_resid

    n_outliers = outlier_mask.sum()
    outlier_text = []
    outlier_text.append(f"Initial fit: {len(y)} observations")
    outlier_text.append(f"Outliers detected: {n_outliers}")

    # Refit without outliers if any found
    if n_outliers > 0:
        clean_y = y[~outlier_mask]
        clean_X = X[~outlier_mask]
        model = sm.OLS(clean_y, clean_X).fit()
        outlier_text.append(f"Refitted with {len(clean_y)} observations")
    else:
        outlier_text.append("No outliers detected, using original fit")

    # Get diagnostics
    fitted_values = model.fittedvalues
    residuals = model.resid

    # Save diagnostic plots if path provided
    if diagnostic_path:
        save_diagnostic_plots(fitted_values, residuals, diagnostic_path)

    summary_str = str(model.summary())

    return model, outlier_text, summary_str, fitted_values, residuals


def save_diagnostic_plots(fitted, resid, filepath):
    """Save diagnostic plots to file."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    try:
        # Convert to numpy arrays
        fitted_vals = np.array(fitted)
        resid_vals = np.array(resid)

        # Try to import seaborn, use basic plots if not available
        try:
            from statsmodels.graphics.gofplots import qqplot
            import seaborn as sns
            use_seaborn = True
        except ImportError:
            use_seaborn = False

        # QQ Plot
        if use_seaborn:
            from statsmodels.graphics.gofplots import qqplot
            qqplot(resid_vals, line='s', ax=axes[0], markersize=3)
        else:
            stats.probplot(resid_vals, dist="norm", plot=axes[0])
        axes[0].set_title('Residual Q-Q Plot')

        # Residual plot
        if use_seaborn:
            import seaborn as sns
            sns.residplot(x=fitted_vals, y=resid_vals, lowess=True,
                          line_kws={'color': 'red', 'lw': 1},
                          scatter_kws={'alpha': 0.4, 's': 10}, ax=axes[1])
        else:
            axes[1].scatter(fitted_vals, resid_vals, alpha=0.4, s=10)

        axes[1].axhline(0, color='grey', linewidth=1)
        axes[1].set_xlabel('Fitted')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Fitted vs. Residuals')

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Could not save diagnostic plots: {e}")
        import traceback
        traceback.print_exc()


def custom_predict(data, model, rseed=None):
    """
    Make predictions for missing response values.

    Returns:
        predictions: array of predictions
        standard_errors: array of standard errors
    """
    if rseed is not None:
        np.random.seed(rseed)

    # For simplicity, predict on all data
    # In practice, you'd only predict where response is missing
    predictions = model.predict(model.model.exog)

    # Get prediction standard errors
    prediction_results = model.get_prediction(model.model.exog)
    standard_errors = prediction_results.se_mean

    return predictions, standard_errors


# scrollable overall frame (To be used in Interactive  Model Builder)
class ScrollableFrame(ttk.Frame):  # overall frame
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


# Interactive Model builder
class OLSPromptGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Impute from Model — interactive model builder')  # title
        self.geometry('1100x700')  # size

        container = ScrollableFrame(self)  # scrollable y for the whole window
        container.pack(fill="both", expand=True)
        frame = container.scrollable_frame

        # initialize variables
        self.data = None
        self.response_var = None
        self.predictors = []
        self.formula_str = ''
        self.pred_rseed = 1

        ###### UI ############
        # top frame to load CSV
        top = ttk.Frame(frame, padding=8)
        top.pack(side='top', fill='x')
        top.columnconfigure(1, weight=1)

        loadcsv_btn = ttk.Button(top, text='Load CSV', command=self.load_csv)
        loadcsv_btn.grid(row=0, column=0, pady=5, padx=5, sticky="w")

        self.status_label = ttk.Label(top, text="Waiting for command.")
        self.status_label.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        self.progress = ttk.Progressbar(top, mode="indeterminate", length=150)
        self.progress.grid(row=0, column=3, padx=5, pady=5, sticky="e")

        rseedlab = ttk.Label(top, text="Random Seed (for replicability):")
        rseedlab.grid(row=1, column=0, pady=5, padx=5, sticky="w")
        self.pred_rseed_var = tk.IntVar(value=1)
        rseedentry = ttk.Entry(top, textvariable=self.pred_rseed_var)
        rseedentry.grid(row=1, column=1, pady=5, padx=5, sticky="w")

        # middle section of UI (controls is user input, output is model results)
        middle = ttk.Frame(frame)
        middle.pack(fill='both', expand=True)

        controls = ttk.Frame(middle, padding=8)
        controls.pack(side='left', fill='y')

        ## specify model section
        ttk.Label(controls,
                  text='Prediction model formula\n(Use step-by-step builder or formula format guide below)').pack(
            anchor='w')

        link_label = tk.Label(
            controls, text='https://www.statsmodels.org/stable/example_formulas.html',
            fg="blue", cursor="hand2", font=('Arial', 9, 'underline')
        )
        link_label.pack(anchor="w", padx=10)
        link_label.bind(
            "<Button-1>",
            lambda e: webbrowser.open_new("https://www.statsmodels.org/stable/example_formulas.html")
        )

        self.direct_formula_text = tk.Text(controls, height=3, width=40)
        self.direct_formula_text.pack()
        ttk.Button(controls, text='Use direct model formula', command=self.use_direct_formula).pack(fill='x', pady=4)
        # Button to open step-by-step popout
        ttk.Button(controls, text="Open step-by-step model formula builder", command=self.open_step_builder).pack(
            anchor='w', pady=(6, 0))

        ttk.Separator(controls).pack(fill='x', pady=6)

        ## outlier detection and removal thresholds - Fixed grid layout
        threshold_frame = ttk.Frame(controls)
        threshold_frame.pack(fill='x', pady=5)

        ttk.Label(threshold_frame, text='Diagnostics / thresholds').grid(row=0, column=0, columnspan=2, sticky='w',
                                                                         pady=5)
        ttk.Label(threshold_frame, text="Cook's Distance Threshold").grid(row=1, column=0, pady=5, padx=5, sticky="w")
        self.cooks_var = tk.DoubleVar(value=1.0)
        ttk.Entry(threshold_frame, textvariable=self.cooks_var).grid(row=1, column=1, pady=5, padx=5, sticky="w")
        ttk.Label(threshold_frame, text='Studentized Residual Threshold').grid(row=2, column=0, pady=5, padx=5,
                                                                               sticky="w")
        self.outlier_var = tk.DoubleVar(value=3.0)
        ttk.Entry(threshold_frame, textvariable=self.outlier_var).grid(row=2, column=1, pady=5, padx=5, sticky="w")

        self.diagnostic_folder = tk.StringVar()
        ttk.Button(controls, text='Choose diagnostic plot save location (optional)',
                   command=self.choose_diag_folder).pack(fill='x', pady=4)
        ttk.Label(controls, textvariable=self.diagnostic_folder, wraplength=200).pack()

        self.fit_btn = ttk.Button(controls, text='Fit model now', command=self.processing_fitting, state='disabled')
        self.fit_btn.pack(fill='x', pady=8)

        ## Outputs to the right of controls
        outputs = ttk.Frame(middle, padding=8)
        outputs.pack(side='left', fill='both', expand=True)

        ttk.Label(outputs, text='Constructed formula:').pack(anchor='w')
        self.formula_label = ttk.Label(outputs, text='', wraplength=700, background='white')
        self.formula_label.pack(fill='x', pady=4)

        ## where the model summary will appear
        ttk.Label(outputs, text='Model Summary (after outlier removal)').pack(anchor='w')
        self.summary_text = tk.Text(outputs, height=12)
        self.summary_text.pack(fill='both', expand=False)

        ## diagnostic plots appear here
        self.plot_frame = ttk.Frame(outputs)
        self.plot_frame.pack(fill='both', expand=True)

        # Fixed save location grid
        save_frame = ttk.Frame(outputs)
        save_frame.pack(fill='x', pady=5)
        ttk.Label(save_frame, text="Data save location with imputed missing response:").grid(row=0, column=0,
                                                                                             sticky=tk.W, padx=5,
                                                                                             pady=3)
        self.final_save = tk.StringVar(value="")
        ttk.Entry(save_frame, textvariable=self.final_save, width=40).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(save_frame, text="Browse", command=lambda: self.browse_path(self.final_save)).grid(row=0, column=2,
                                                                                                      padx=5)

        self.proceed_btn = ttk.Button(outputs, text='Proceed to custom predictions', command=self.processing_prediction,
                                      state='disabled')
        self.proceed_btn.pack(fill='x', pady=6)

        # initialize
        self.pred_config = {}
        self.interactions = []
        self.fitted_model = None
        self.prev_step_state = {}  # store previous step builder state

    def browse_path(self, var):
        """Browse for file save location."""
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            var.set(path)

    def update_fit_button_state(self):
        """disable fit button if no model formula is supplied"""
        if self.formula_str:
            self.fit_btn.config(state='normal')
        else:
            self.fit_btn.config(state='disabled')

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if not path:
            return
        df = pd.read_csv(path, index_col=0, low_memory=False)

        # Get column names
        cols = df.columns.tolist()

        # Ask for response variable
        resp = simpledialog.askstring('Response variable',
                                      f'Columns found:\n{", ".join(cols[:10])}\n{"..." if len(cols) > 10 else ""}\n\nEnter the response variable:')
        if resp not in cols:
            messagebox.showerror('Error', f'{resp} not in columns')
            return

        # Subset data to just possible model variables
        possible_vars = possible_variables(df, resp)
        possible_vars.append(resp)
        self.data = df[possible_vars]
        self.response_var = resp

        messagebox.showinfo("Success",
                            f"Loaded data with {len(self.data)} rows and {len(possible_vars)} columns.\nResponse variable: {resp}")

    def check_transform(self, varstr):
        """Check if a variable or transformation exists in the data."""
        if varstr in self.data.columns:
            return varstr
        else:
            if "poly" in varstr:
                internalvar = varstr.replace("poly", "").replace("(", "")
                internalvar_split = internalvar.split(",")
                internalvar = internalvar_split[0].strip()
                if internalvar in self.data.columns:
                    return varstr
        return None

    def use_direct_formula(self):
        if self.data is None or self.response_var is None:
            messagebox.showwarning('No data', 'Load data first')
            return
        text = self.direct_formula_text.get('1.0', 'end').strip()
        if not text:
            messagebox.showwarning('Empty', 'Enter a formula')
            return

        # Basic validation - just check that formula has ~ separator
        if '~' not in text:
            messagebox.showwarning('Invalid formula', 'Formula must contain ~ separator')
            return

        self.formula_str = text
        self.formula_label.config(text=self.formula_str)
        self.update_fit_button_state()

    def open_step_builder(self):
        """Open the interactive variable selection dialog."""
        if self.data is None or self.response_var is None:
            messagebox.showwarning('No data', 'Load data first')
            return

        dialog = VariableSelectionDialog(self, self.data, self.response_var, self.prev_step_state)
        self.wait_window(dialog)

        if dialog.formula_str:
            # Save user selections for next time
            self.formula_str = dialog.formula_str
            self.prev_step_state = dialog.save_state()

            # Update the main GUI textboxes and labels
            self.formula_label.config(text=self.formula_str)
            self.direct_formula_text.delete('1.0', 'end')
            self.direct_formula_text.insert('1.0', self.formula_str)
            self.update_fit_button_state()

    def choose_diag_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.diagnostic_folder.set(d)

    def fit_model(self):
        if not self.formula_str:
            messagebox.showwarning('No formula', 'Build or enter a formula first')
            return
        if self.data is None:
            messagebox.showwarning('No data', 'Load data first')
            return

        try:
            cooks = float(self.cooks_var.get())
            outthr = float(self.outlier_var.get())
            diag = None
            if self.diagnostic_folder.get():
                diag = os.path.join(self.diagnostic_folder.get(), 'diagnostic_plots.png')

            model, outliertext, summaryobj, fittedvals, residvals = get_model(
                self.data, self.formula_str, cooks, outthr, diag, True, True
            )

            # Store model and output text safely
            self.fitted_model = model
            summary_info = (model, outliertext, summaryobj, fittedvals, residvals)

            self.after(0, lambda: self.update_after_fit(summary_info))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror('Model error', f'Error fitting model: {e}'))
            self.after(0, self.done_processing_fit)

    def _show_diagnostic_plots(self, summary_info):
        model, outliertext, summaryobj, fitted, resid = summary_info
        # create figure similar to save_diagnostic_plots
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.3))

        try:
            # Convert to numpy arrays if needed
            fitted_vals = np.array(fitted)
            resid_vals = np.array(resid)

            # Try importing required libraries
            try:
                from statsmodels.graphics.gofplots import qqplot
                import seaborn as sns
                use_seaborn = True
            except ImportError:
                use_seaborn = False
                print("Seaborn not available, using basic plots")

            # Create QQ plot
            if use_seaborn:
                from statsmodels.graphics.gofplots import qqplot
                qqplot(resid_vals, line='s', ax=axes[0], markersize=1)
            else:
                # Basic QQ plot alternative
                from scipy import stats
                stats.probplot(resid_vals, dist="norm", plot=axes[0])
            axes[0].set_title('Residual Q-Q Plot')

            # Create residual plot
            if use_seaborn:
                import seaborn as sns
                sns.residplot(x=fitted_vals, y=resid_vals, lowess=True,
                              line_kws={'color': 'red', 'lw': 1},
                              scatter_kws={'alpha': 0.4, 's': 1}, ax=axes[1])
            else:
                # Basic scatter plot
                axes[1].scatter(fitted_vals, resid_vals, alpha=0.4, s=1)
                # Add a horizontal line at 0
                axes[1].axhline(0, color='grey', linewidth=1, linestyle='--')

            axes[1].axhline(0, color='grey', linewidth=1)
            axes[1].set_xlabel('Fitted')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title('Fitted vs. Residuals')

            for i in [0, 1]:
                axes[i].tick_params(axis="both", which="major", labelsize=8)
                axes[i].tick_params(axis="both", which="minor", labelsize=6)

            # Apply tight layout after plots are created
            fig.tight_layout()

        except Exception as e:
            # Clear the axes and show error message
            import traceback
            error_msg = f'Could not create plots:\n{str(e)}\n\n{traceback.format_exc()}'
            print(error_msg)  # Print to console for debugging

            for ax in axes:
                ax.clear()
            axes[0].text(0.5, 0.5, f'Could not create plots:\n{str(e)}',
                         ha='center', va='center', wrap=True, fontsize=9)
            axes[0].set_xlim(0, 1)
            axes[0].set_ylim(0, 1)
            axes[0].axis('off')
            axes[1].axis('off')
            fig.tight_layout()

        # Clear previous plots
        for child in self.plot_frame.winfo_children():
            child.destroy()

        # Embed the figure in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def processing_prediction(self):
        if not self.fitted_model:
            return
        self.progress.start(10)
        self.status_label.config(text="Predicting missing " + self.response_var + " values from model...")
        threading.Thread(target=self.predict_response, daemon=True).start()

    def processing_fitting(self):
        self.progress.start(10)
        self.status_label.config(text="Fitting model...")
        threading.Thread(target=self.fit_model, daemon=True).start()

    def update_after_fit(self, summary_info):
        """Runs in main thread to update GUI after fitting."""
        model, outliertext, summaryobj, fittedvals, residvals = summary_info

        # Update summary text
        try:
            self.summary_text.delete('1.0', 'end')
            for line in outliertext:
                self.summary_text.insert('end', line + "\n")
            self.summary_text.insert('end', "\n" + summaryobj)
        except Exception:
            self.summary_text.insert('1.0', 'Could not render model summary.')

        # Show diagnostic plots (this must run on main thread)
        try:
            self._show_diagnostic_plots(summary_info)
        except Exception as e:
            print(f"Could not show diagnostic plots: {e}")

        # Enable proceed button and stop spinner
        self.proceed_btn.config(state='normal')
        self.done_processing_fit()

    def done_processing_fit(self):
        self.progress.stop()
        self.status_label.config(text="Done fitting model.")

    def done_processing_pred(self):
        self.progress.stop()
        self.status_label.config(text="Done predicting.")

    def predict_response(self):
        try:
            pred, se = custom_predict(self.data, self.fitted_model, self.pred_rseed_var.get())
            outdf = self.data.copy()
            outdf['prediction'] = pred
            outdf['se_fit'] = se

            # Save if path specified
            if self.final_save.get():
                outdf.to_csv(self.final_save.get())
                self.after(0, lambda: messagebox.showinfo("Success", f"Predictions saved to {self.final_save.get()}"))

            # Show preview
            self.after(0, lambda: PreviewWindow(self, outdf))
            self.after(0, self.done_processing_pred)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror('Prediction error', f'Error making predictions: {e}'))
            self.after(0, self.done_processing_pred)


# =============================================================
# Step-by-step Variable Selection Dialog
# =============================================================
class VariableSelectionDialog(tk.Toplevel):
    """Popup dialog to configure variables, transformations, and interactions."""

    def __init__(self, parent, df, response_var, prev_state):
        super().__init__(parent)
        self.parent = parent
        self.df = df
        self.response_var = response_var
        self.prev_state = prev_state or {}
        self.formula_str = None

        self.title("Step-by-step Model Builder")
        self.geometry("850x700")

        # State holders
        self.variable_settings = {}
        self.selected_interactions = self.prev_state.get('interactions', [])
        self.response_transform = tk.StringVar(value=self.prev_state.get('response_transform', 'none'))

        # Scrollable container
        container = ttk.Frame(self)
        canvas = tk.Canvas(container)
        scrollbar_y = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set)
        container.pack(fill="both", expand=True)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_y.pack(side="right", fill="y")

        self.scrollable_frame = scrollable_frame
        self._build_interface()

    def _build_interface(self):
        """Builds and optionally restores all variable widgets."""
        frame = self.scrollable_frame

        ttk.Label(frame, text=f"Response variable: {self.response_var}", font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=10
        )

        ttk.Label(frame, text="Transform response:").grid(row=1, column=0, sticky="w")
        ttk.OptionMenu(frame, self.response_transform, self.response_transform.get(), "none", "log", "sqrt").grid(row=1,
                                                                                                                  column=1,
                                                                                                                  sticky="w")

        ttk.Separator(frame, orient="horizontal").grid(row=2, column=0, columnspan=5, sticky="ew", pady=10)

        # Header
        ttk.Label(frame, text="Variable").grid(row=3, column=0, padx=5)
        ttk.Label(frame, text="Include").grid(row=3, column=1)
        ttk.Label(frame, text="Transform").grid(row=3, column=2)
        ttk.Label(frame, text="Polynomial degree").grid(row=3, column=3)

        # Variables
        predictor_vars = [c for c in self.df.columns if c != self.response_var]
        for i, var in enumerate(predictor_vars, start=4):
            include_var = tk.BooleanVar(value=False)
            transform_var = tk.StringVar()
            degree_var = tk.StringVar(value="")

            # Default transformation — categorical for non-numeric variables
            if pd.api.types.is_numeric_dtype(self.df[var]):
                transform_var.set("none")
            else:
                transform_var.set("categorical")

            # Restore from previous session if present
            if 'variable_settings' in self.prev_state and var in self.prev_state['variable_settings']:
                prev_cfg = self.prev_state['variable_settings'][var]
                include_var.set(prev_cfg['include'])
                transform_var.set(prev_cfg['transform'])
                degree_var.set(prev_cfg['degree'])

            ttk.Label(frame, text=var).grid(row=i, column=0, sticky="w", padx=5)
            ttk.Checkbutton(frame, variable=include_var).grid(row=i, column=1, padx=5)
            transform_menu = ttk.OptionMenu(frame, transform_var, transform_var.get(), "none", "categorical", "log",
                                            "sqrt")
            transform_menu.grid(row=i, column=2, padx=5)
            poly_entry = ttk.Entry(frame, textvariable=degree_var, width=10)
            poly_entry.grid(row=i, column=3, padx=5)

            # Disable degree if categorical
            def update_poly_state(var_ref=transform_var, entry_ref=poly_entry, degree_ref=degree_var):
                if var_ref.get() == "categorical":
                    entry_ref.config(state="disabled")
                    degree_ref.set("")
                else:
                    entry_ref.config(state="normal")

            transform_var.trace_add("write",
                                    lambda *_, tv=transform_var, pe=poly_entry, pd=degree_var: update_poly_state(tv, pe,
                                                                                                                 pd))
            update_poly_state(transform_var, poly_entry, degree_var)

            self.variable_settings[var] = {"include": include_var, "transform": transform_var, "degree": degree_var}

        next_row = len(predictor_vars) + 5
        ttk.Button(frame, text="Edit interaction terms", command=self._open_interactions_window).grid(
            row=next_row, column=0, columnspan=3, pady=10
        )
        ttk.Button(frame, text="Build formula", command=self._build_formula).grid(
            row=next_row + 1, column=0, columnspan=3, pady=10
        )

    def _open_interactions_window(self):
        """Open dialog for selecting two-way interaction terms."""
        # Get base selected variables (no transform or categorical only)
        base_selected_vars = [v for v, vals in self.variable_settings.items()
                              if vals["include"].get() and vals['transform'].get() in ["none", "categorical"]]

        # Get transformed variables
        transform_vars = []
        for v, vals in self.variable_settings.items():
            if vals['include'].get():
                trans = vals['transform'].get()
                if trans in ["sqrt", "log"]:
                    transform_vars.append(f"{trans}_{v}")

        selected_vars = base_selected_vars + transform_vars

        if not selected_vars or len(selected_vars) < 2:
            messagebox.showwarning("Not enough variables", "Select at least 2 variables before adding interactions.")
            return

        inter_popup = tk.Toplevel(self)
        inter_popup.title("Add Interaction Terms")
        inter_popup.geometry("400x400")

        ttk.Label(inter_popup, text="Select interaction terms:", font=("Arial", 10, "bold")).pack(pady=10)

        # Scrollable frame for interactions
        canvas = tk.Canvas(inter_popup)
        scrollbar = ttk.Scrollbar(inter_popup, orient="vertical", command=canvas.yview)
        inter_frame = ttk.Frame(canvas)
        inter_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inter_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        interaction_vars = []
        for i in range(len(selected_vars)):
            for j in range(i + 1, len(selected_vars)):
                pair = f"{selected_vars[i]}:{selected_vars[j]}"
                var_check = tk.BooleanVar(value=pair in self.selected_interactions)
                ttk.Checkbutton(inter_frame, text=pair, variable=var_check).pack(anchor="w")
                interaction_vars.append((pair, var_check))

        def save_interactions():
            self.selected_interactions = [p for p, var in interaction_vars if var.get()]
            inter_popup.destroy()

        ttk.Button(inter_popup, text="Save Interactions", command=save_interactions).pack(pady=10)

    def _build_formula(self):
        """Construct final formula string in R-style, add transformed columns to data."""
        resp = self.response_var
        rtrans = self.response_transform.get()

        if rtrans == "log":
            lhs = f"np.log({resp})"
        elif rtrans == "sqrt":
            lhs = f"np.sqrt({resp})"
        else:
            lhs = resp

        rhs_terms = []
        transformed_vars = {}  # maps original -> new transformed name

        for var, vals in self.variable_settings.items():
            if vals["include"].get():
                trans = vals["transform"].get()
                deg = vals["degree"].get()

                transformed_name = var  # by default, no change

                # --- Handle transformations and create new columns if needed ---
                if trans == "log":
                    new_name = f"log_{var}"
                    if new_name not in self.df.columns:
                        self.df[new_name] = np.log(self.df[var])
                    transformed_name = new_name
                elif trans == "sqrt":
                    new_name = f"sqrt_{var}"
                    if new_name not in self.df.columns:
                        self.df[new_name] = np.sqrt(self.df[var])
                    transformed_name = new_name

                transformed_vars[var] = transformed_name

                # --- Build RHS term ---
                if trans == "categorical":
                    rhs_terms.append(f"C({var})")
                elif deg.isdigit() and int(deg) > 1:
                    rhs_terms.append(f"poly({transformed_name},{deg})")
                else:
                    rhs_terms.append(transformed_name)

        # --- Handle interactions using transformed variable names ---
        for inter in self.selected_interactions:
            # Parse interaction - handle transformed variable names
            a, b = inter.split(':')

            # Check if it's a transformed variable
            if a.startswith(('log_', 'sqrt_')):
                a_name = a
            else:
                a_name = transformed_vars.get(a, a)

            if b.startswith(('log_', 'sqrt_')):
                b_name = b
            else:
                b_name = transformed_vars.get(b, b)

            rhs_terms.append(f"{a_name}:{b_name}")

        # --- Final formula assembly ---
        if rhs_terms:
            formula = f"{lhs} ~ {' + '.join(rhs_terms)}"
        else:
            messagebox.showwarning("No variables selected", "Please select at least one predictor variable.")
            return

        self.formula_str = formula
        self.destroy()

    def save_state(self):
        """Save the dialog's current state for reuse."""
        return {
            "response_transform": self.response_transform.get(),
            "variable_settings": {
                var: {
                    "include": vals["include"].get(),
                    "transform": vals["transform"].get(),
                    "degree": vals["degree"].get(),
                }
                for var, vals in self.variable_settings.items()
            },
            "interactions": self.selected_interactions,
        }


class PreviewWindow(tk.Toplevel):
    def __init__(self, parent, df):
        super().__init__(parent)
        self.title('Predictions preview')
        self.geometry("800x600")

        # Add scrollbars
        frame = ttk.Frame(self)
        frame.pack(fill='both', expand=True)

        text = tk.Text(frame, height=20, wrap='none')

        vsb = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        text.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        text.insert('1.0', df.head(20).to_string())
        text.config(state='disabled')


if __name__ == '__main__':
    app = OLSPromptGUI()
    app.mainloop()