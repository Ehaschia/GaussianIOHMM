from io_module.common import UNK


class UNKRefiner:
    def __init__(self, level=5, alphabet=None):
        self.level = level
        self.alphabet = alphabet

    def refine(self, word, pos):
        signature = UNK
        if self.level == 5:
            # TODO in English no difference between Uppercase and Titlecase.
            # we need to consider other language
            ncap = 0
            wlen = len(word)
            hasDash = False
            hasDigit = False
            hasLower = False

            for i in range(wlen):
                ch = word[i]
                if ch.isdigit():
                    hasDigit = True
                elif ch == '-':
                    hasDash = True
                elif ch.isalpha():
                    if ch.islower():
                        hasLower = True
                    else:
                        ncap += 1
            ch = word[0]
            lower = word.lower()
            if ch.isupper():
                if pos == 0 and ncap == 1:
                    signature += '-INITC'
                    if self.alphabet.is_known(lower):
                        signature += '-KNOWNLC'
                else:
                    signature += '-CAPS'
            elif (not ch.isalpha()) and ncap > 1:
                signature += '-CAPS'
            elif hasLower:
                signature += '-LC'
            if hasDash: signature += '-DASH'
            if hasDigit: signature += '-NUM'

            if lower.endswith('s') and wlen > 3:
                ch = lower[wlen - 2]
                if ch != 's' and ch != 'i' and ch != 'u':
                    signature += '-s'
            elif wlen > 5 and not hasDash and not (hasDigit and ncap > 0):
                if lower.endswith('ed'):
                    signature += '-ed'
                elif lower.endswith('ing'):
                    signature += '-ing'
                elif lower.endswith('ion'):
                    signature += '-ion'
                elif lower.endswith('er'):
                    signature += '-er'
                elif lower.endswith('est'):
                    signature += '-est'
                elif lower.endswith('ly'):
                    signature += '-ly'
                elif lower.endswith('ity'):
                    signature += '-ity'
                elif lower.endswith('y'):
                    signature += '-y'
                elif lower.endswith('al'):
                    signature += '-al'
        elif self.level == 4:
            hasDash = False
            hasDigit = False
            hasLower = False
            hasComma = False
            hasLetter = False
            hasPeriod = False
            hasNonDigit = False


            for ch in word:
                if ch.isdigit():
                    hasDigit = True
                else:
                    hasNonDigit = True
                    if ch.isalpha():
                        hasLetter = True
                        if ch.islower():
                            hasLower = True
                    else:
                        if ch == '-':
                            hasDash = True
                        elif ch == '.':
                            hasPeriod = True
                        elif ch == ',':
                            hasComma = True
            if word[0].isupper():
                if not hasLower:
                    signature += '-AC'
                elif pos == 0:
                    signature += '-SC'
                else:
                    signature += '-C'
            elif hasLower:
                signature += 'L'
            elif hasLetter:
                signature += '-U'
            else:
                signature += '-S'

            if hasDigit and not hasNonDigit:
                signature += '-N'
            elif hasDigit:
                signature += '-n'

            if hasDash: signature += '-D'
            if hasComma: signature += '-C'
            if hasPeriod: signature += '-P'

            if len(word) > 3:
                ch = word[-1]
                if ch.isalpha():
                    signature += '-' + ch.lower()
        elif self.level == 3:
            signature += '-'
            num = 0
            newClass = '-'
            lastClass = '-'

            for ch in word:
                if ch.isupper():
                    if pos == 0:
                        newClass = 'S'
                    else:
                        newClass = 'L'
                elif ch.isalpha():
                    newClass = 'l'
                elif ch.isdigit():
                    newClass = 'd'
                elif ch == '-':
                    newClass = 'h'
                elif ch == '.':
                    newClass = 'p'
                else:
                    newClass = 's'

                if newClass != lastClass:
                    lastClass = newClass
                    signature += lastClass
                    num = 1
                else:
                    if num < 2:
                        signature += '+'
                    num += 1
            if len(word) > 3:
                ch = word[-1].lower()
                signature += '-' + ch
        elif self.level == 2:
            hasNoDigit = False
            hasDigit = False
            hasLower = False

            for ch in word:
                if ch.isdigit():
                    hasDigit = True
                else:
                    hasNoDigit = True
                    if ch.isalpha():
                        if ch.islower():
                            hasLower = True
            if word[0].upper():
                if not hasLower:
                    signature += '-ALLC'
                elif pos == 0:
                    signature += '-INIT'
                else:
                    signature += '-UC'
            elif hasLower:
                signature += '-LC'
            if word.find('-') != -1:
                signature += '-DASH'
            if hasDigit:
                if not hasNoDigit:
                    signature += '-NUM'
                else:
                    signature += '-DIG'
            elif len(word) > 3:
                ch = word[-1].lower()
                signature += '-' + ch
        elif self.level == 1:
            signature += '-'
            signature += word[max(len(word) - 2, 0):]
            signature += '-'
            if word[0].islower():
                signature += 'LOWER'
            else:
                if pos == 0:
                    signature += 'INIT'
                else:
                    signature += 'UPPER'
        else:
            pass

        return signature
