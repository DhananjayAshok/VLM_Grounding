

def fix_string(string):
    if not isinstance(string, str):
        return string
    stop_string = "[STOP]"
    for i in range(len(stop_string), -1, -1):
        if stop_string[:i] in string:
            string = string.replace(stop_string[:i], "")
    return string

def fix_df(df, columns):
    for column in columns:
        if column in ["full_information_question", "image_reference_question"]:
            continue
        df[column] = df[column].apply(fix_string)
    return df