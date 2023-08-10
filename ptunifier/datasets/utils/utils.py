import re


def split_into_sents(text):
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|Fig|FIG|fig|Figs|figs|Figure|FIGURE|figure|figures|Figures|al|No|\d)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|\d)"

    text_clone = re.sub("[Ff][Ii][Gg][Uu]*[Rr]*[Ee]*[\s]*[.]*[\s]*[\d]+[\s]*[.]*[\s]*[:]*[-]*[\s]*", "", text)
    text_clone = text_clone.replace(".)", ")")
    text_clone = " " + text_clone + "  "
    text_clone = text_clone.replace("\n", " ")
    text_clone = re.sub(prefixes, "\\1<prd>", text_clone)
    text_clone = re.sub(websites, "<prd>\\1", text_clone)
    if "Ph.D" in text_clone: text_clone = text_clone.replace("Ph.D.", "Ph<prd>D<prd>")
    text_clone = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text_clone)
    text_clone = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text_clone)
    text_clone = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>",
                        text_clone)
    text_clone = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text_clone)
    text_clone = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text_clone)
    text_clone = re.sub(" " + suffixes + "[.]", " \\1<prd>", text_clone)
    text_clone = re.sub(" " + alphabets + "[.]", " \\1<prd>", text_clone)
    if "”" in text_clone: text_clone = text_clone.replace(".”", "”.")
    if "\"" in text_clone: text_clone = text_clone.replace(".\"", "\".")
    if "!" in text_clone: text_clone = text_clone.replace("!\"", "\"!")
    if "?" in text_clone: text_clone = text_clone.replace("?\"", "\"?")
    text_clone = text_clone.replace(".", ".<stop>")
    text_clone = text_clone.replace("?", "?<stop>")
    text_clone = text_clone.replace("!", "!<stop>")
    text_clone = text_clone.replace("<prd>", ".")
    sentences = text_clone.split("<stop>")
    sentences = [s.strip() for s in sentences if s.strip() != ""]
    return sentences
