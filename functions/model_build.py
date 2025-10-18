
## Find variables that can be used to impute missing response entries.
# i.e. variables that are never missing when the response is and that take more than 1 unique value.
def possible_variables(data,response):
    naidx=data[response].isna() #missing response
    missing_y=data[naidx] #subset of data where response is missing
    possible_vars=list() #initialize
    for cvar in data.columns: #for each column
        if cvar==response: #skip response variable
            pass
        else:
            if missing_y[cvar].isna().sum()==0: #none missing when response is missing
                if data[cvar].nunique()>1: #more than 1 unique value
                    possible_vars.append(cvar) #add to list of possible model variables
    return possible_vars
