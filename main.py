"""
Programming practice tool
"""
import ast
import numpy as np
import random
import string
import os
import rlcompleter
import readline
import itertools


def load_corpus(d):
    """Load corpus from a directory

    Parameters
    ----------
    d : string
        The directory to load from. Corpus files should be flatfiles
        inside this directory, with one expression per line.

    Returns
    -------
    corpus : dict
        Dict mapping expression type to a list of expression templates.
    """
    corpus = dict()
    for fname in os.listdir(d):
        if fname.endswith('~'):
            continue
        f = open(os.path.join(d, fname))
        lines = [x.strip() for x in f.readlines()]
        corpus[fname] = lines
    return corpus


def safe_eval(s):
    """Eval a string, catching exceptions.

    Parameters
    ----------
    s : string
        String to eval

    Returns
    -------
    val : Python value
        Either the value of the expression, or the exception that was
        raised when trying to evaluate the expression.
    """
    try:
        return eval(s)
    except Exception as e:
        return e


def safe_compile(s):
    """Compile a string to an abstract syntax tree, catching
    exceptions.

    Parameters
    ----------
    s : string
        String to eval

    Returns
    -------
    val : ast.Expression
        Either the AST of the expression, or the AST of the exception
        that was raised when trying to evaluate the expression.
    """
    try:
        x = compile(s, 'NO FILE', 'eval', ast.PyCF_ONLY_AST)
        return x
    except Exception as e:
        return compile('e', 'NO FILE', 'eval',
                       ast.PyCF_ONLY_AST)


def instantiate(expr, corpus):
    """Instantiate an expression template from the corpus.

    Parameters
    ----------
    expr : string
        Expression template that will be instantiated.
    corpus : dict
        Corpus in the formate returned by load_corpus(), used to fill
        in placeholders in the expression.

    Returns
    -------
    expression : string
        A fully instantiated expression, ready for evaluation.
    """
    # remember the original expression in case we decide to abort and
    # try again
    original_expr = expr

    f = string.Formatter()

    # repeat while the expression isn't fully instantiated
    while True:
        next_expr = ''
        for s, var, fmt, conv in f.parse(expr):
            assert conv is None
            # replace { with double {{ and } with double }}
            next_expr += s.replace('{', '{{').replace('}', '}}')

            # if there is something that needs to be filled in
            if var is not None:

                # if there's a non-empty fmt string then we're binding
                # a variable
                if fmt:
                    bound_expr = ''
                    variable_expr = random.choice(corpus[var])
                    variable_type, variable_name = fmt.split(':')
                    for s2, var2, fmt2, conv2 in f.parse(variable_expr):
                        # replace { with double {{ and } with double }}
                        bound_expr += s2.replace('{', '{{').replace('}', '}}')
                        assert conv2 is None
                        if var2 is not None:
                            if var2 != variable_type:
                                # weird case, abort and reroll
                                return instantiate(original_expr, corpus)
                            else:
                                bound_expr += variable_name
                    next_expr += bound_expr

                # we're not binding a variable, so just append a
                # random corpus template of the right type
                else:
                    next_expr += random.choice(corpus[var])

        expr = next_expr

        # check if we're done instantiating the template
        stuff = list(f.parse(expr))
        if all(x[1] is None for x in stuff):
            break

    # replace double {{ with { and double }} with }
    expr = expr.replace('{{', '{').replace('}}', '}')
    return expr


class Problem(object):
    """Abstract base class for problems"""
    pass


class SingletonForwardProblem(Problem):
    """Abstract base class for problems whose answer should be a
    single value of a given type"""
    _type = None

    def check_answer(self, answer):
        """Check a user-provided answer.
        
        Parameters
        ----------
        answer : string
            The answer to be checked.

        Returns
        -------
        passed : bool
            Whether the user successfully passed the test.
        msg : string
            Message to print.
        """
        if safe_eval(answer) == self.answer:
            x = safe_compile(answer)
            assert self._type is not None
            if isinstance(x.body, self._type):
                return True, 'Correct!'
            else:
                return False, 'Correct, but not fully simplified'
        else:
            return False, 'Incorrect :('


class BoolForwardProblem(Problem):
    """Abstract base class for problems whose answer should be a
    single bool"""
    def check_answer(self, answer):
        """Check a user-provided answer.
        
        Parameters
        ----------
        answer : string
            The answer to be checked.

        Returns
        -------
        passed : bool
            Whether the user successfully passed the test.
        msg : string
            Message to print.
        """
        if safe_eval(answer) == self.answer:
            x = safe_compile(answer)
            if isinstance(x.body, ast.Name) and x.body.id in ['True', 'False']:
                return True, 'Correct!'
            else:
                return False, 'Correct, but not fully simplified'
        else:
            return False, 'Incorrect :('


class ListOfSameForwardProblem(Problem):
    """Abstract base class for problems whose answer should be a list of
    elements of the same type"""
    _element_type = None

    def check_answer(self, answer):
        """Check a user-provided answer.
        
        Parameters
        ----------
        answer : string
            The answer to be checked.

        Returns
        -------
        passed : bool
            Whether the user successfully passed the test.
        msg : string
            Message to print.
        """
        if safe_eval(answer) == self.answer:
            x = safe_compile(answer)
            if isinstance(x.body, ast.List):
                y = x.body
                assert self._element_type is not None
                for z in y.elts:
                    if not isinstance(z, self._element_type):
                        return False, 'Correct, but not fully simplified'
                return True, 'Correct!'
            else:
                return False, 'Correct, but not fully simplified'
        else:
            return False, 'Incorrect :('


class SetOfSameForwardProblem(Problem):
    """Abstract base class for problems whose answer should be a set of
    elements of the same type"""
    _element_type = None

    def check_answer(self, answer):
        """Check a user-provided answer.
        
        Parameters
        ----------
        answer : string
            The answer to be checked.

        Returns
        -------
        passed : bool
            Whether the user successfully passed the test.
        msg : string
            Message to print.
        """
        if safe_eval(answer) == self.answer:
            x = safe_compile(answer)
            if isinstance(x.body, ast.Set):
                y = x.body
                assert self._element_type is not None
                for z in y.elts:
                    if not isinstance(z, self._element_type):
                        return False, 'Correct, but not fully simplified'
                return True, 'Correct!'
            else:
                return False, 'Correct, but not fully simplified'
        else:
            return False, 'Incorrect :('


class DictOfSameSameForwardProblem(Problem):
    """Abstract base class for problems whose answer should be a dict
    mapping elements of one type to elements of another type"""
    _key_type = None
    _value_type = None

    def check_answer(self, answer):
        """Check a user-provided answer.
        
        Parameters
        ----------
        answer : string
            The answer to be checked.

        Returns
        -------
        passed : bool
            Whether the user successfully passed the test.
        msg : string
            Message to print.
        """
        eval_answer = safe_eval(answer)
        if (isinstance(eval_answer, dict) and eval_answer == self.answer):
            x = safe_compile(answer)
            if isinstance(x.body, ast.Dict):
                y = x.body
                assert self._key_type is not None
                assert self._value_type is not None
                for z in y.keys:
                    if not isinstance(z, self._key_type):
                        return False, 'Correct, but not fully simplified'
                for z in y.values:
                    if not isinstance(z, self._value_type):
                        return False, 'Correct, but not fully simplified'
                return True, 'Correct!'
            else:
                return False, 'Correct, but not fully simplified'
        else:
            return False, 'Incorrect :('


class NumForwardProblem(SingletonForwardProblem):
    """Abstract base class for problems whose answer should be a
    number"""
    _type = ast.Num


class StringForwardProblem(SingletonForwardProblem):
    """Abstract base class for problems whose answer should be a
    string"""
    _type = ast.Str


class ListOfNumForwardProblem(ListOfSameForwardProblem):
    """Abstract base class for problems whose answer should be a list
    of numbers"""
    _element_type = ast.Num


class ListOfStringForwardProblem(ListOfSameForwardProblem):
    """Abstract base class for problems whose answer should be a list
    of strings"""
    _element_type = ast.Str


class SetOfNumForwardProblem(SetOfSameForwardProblem):
    """Abstract base class for problems whose answer should be a list
    of strings"""
    _element_type = ast.Num


class CorpusProblem(Problem):
    """Abstract base class for problems which should be populated by
    instantiating a template from the corpus"""
    _seed_type = None

    def __init__(self):
        assert self._seed_type is not None
        if isinstance(self._seed_type, list):
            seed_type = random.choice(self._seed_type)
        else:
            seed_type = self._seed_type
        expr = (random.choice(corpus[seed_type]))
        self.expr = instantiate(expr, corpus)
        self.answer = safe_eval(self.expr)

    def prompt(self):
        """Return the prompt to show the user.

        Returns
        -------
        prompt : string
            The prompt.
        """
        return self.expr


class ListComprehensionBackwardProblem(Problem):
    """Problem which gives you a list of nums and asks you to find a
    list comprehension generating them"""

    def check_answer(self, answer):
        """Check a user-provided answer.
        
        Parameters
        ----------
        answer : string
            The answer to be checked.

        Returns
        -------
        passed : bool
            Whether the user successfully passed the test.
        msg : string
            Message to print.
        """
        if safe_eval(answer) == self.answer:
            x = safe_compile(answer)
            if isinstance(x.body, ast.ListComp):
                gens = x.body.generators
                if any(isinstance(g.iter, ast.List) for g in gens):
                    return False, """
Correct, but you are cheating by using a list inside the list comprehension
"""
                return True, 'Correct!'
            else:
                return False, 'Correct, but not a list comprehension'
        else:
            return False, 'Incorrect :('

    def prompt(self):
        """Return the prompt to show the user.

        Returns
        -------
        prompt : string
            The prompt.
        """
        return 'Find a list comprehension that returns: ' + repr(
            self.answer)


class AdditionProblem(NumForwardProblem):
    """Problem which requires you to add two numbers"""
    def __init__(self):
        self.a = np.random.randint(10)
        self.b = np.random.randint(10)
        self.answer = self.a + self.b

    def prompt(self):
        """Return the prompt to show the user.

        Returns
        -------
        prompt : string
            The prompt.
        """
        return '%s + %s' % (self.a, self.b)


class NumCorpusProblem(CorpusProblem, NumForwardProblem):
    """Problem which requires you to simplify a num expression from
    the corpus"""
    _seed_type = 'num_expr'


class BoolCorpusProblem(CorpusProblem, BoolForwardProblem):
    """Problem which requires you to simplify a bool expression from
    the corpus"""
    _seed_type = 'bool_expr'


class StringCorpusProblem(CorpusProblem, StringForwardProblem):
    """Problem which requires you to simplify a string expression from
    the corpus"""
    _seed_type = 'string_expr'


class ListOfStringCorpusProblem(CorpusProblem, ListOfStringForwardProblem):
    """Problem which requires you to simplify a list of string
    expression from the corpus"""
    _seed_type = 'list_of_string_expr'


class ListOfNumCorpusProblem(CorpusProblem, ListOfNumForwardProblem):
    """Problem which requires you to simplify a list of num expression
    from the corpus"""
    _seed_type = 'list_of_num_expr'


class SetOfNumCorpusProblem(CorpusProblem, SetOfNumForwardProblem):
    """Problem which requires you to simplify a set of num expression
    from the corpus"""
    _seed_type = 'set_comprehension_of_num'


class DictOfStringNumCorpusProblem(CorpusProblem,
                                   DictOfSameSameForwardProblem):
    """Problem which requires you to simplify a dict expression
    mapping strings to nums from the corpus"""
    _seed_type = ['dict_comprehension_of_string_num',
                  'dict_expr_of_string_num']
    _key_type = ast.Str
    _value_type = ast.Num


class ListComprehensionBackwardCorpusProblem(ListComprehensionBackwardProblem,
                                             CorpusProblem):
    """Problem which requires you to produce a list comprehension that
    matches the output of a list comprehension from the corpus"""
    _seed_type = 'list_comprehension_of_num'


def get_problem():
    """Return a random problem instance.

    Returns
    -------
    prob : Problem
    """
    cls = random.choice([
            SetOfNumCorpusProblem,
            BoolCorpusProblem,
            DictOfStringNumCorpusProblem,
            ListOfNumCorpusProblem,
            ListOfStringCorpusProblem,
            StringCorpusProblem,
            NumCorpusProblem,
            ListComprehensionBackwardCorpusProblem,
            ])
    return cls()


def explore_loop():
    """Read-Eval-Print Loop for exploration mode"""
    while True:
        try:
            answer = raw_input('>exp>')
        except EOFError:
            return

        if answer == 'q' or answer == 'quit':
            return

        print repr(safe_eval(answer))


def generate_loop():
    """Mode that generates examples until user hits Ctrl-C"""
    try:
        while True:
            prob = get_problem()
            print prob.prompt()
    except KeyboardInterrupt:
        return


def main_loop():
    """Main loop of the program"""

    # generate the first problem
    prob = get_problem()

    # track number of correct answers
    ncorrect = 0

    while True:
        print
        print prob.prompt()
        try:
            answer = raw_input('>>>')
        except EOFError:
            break

        # go into exploration mode
        if answer == 'e' or answer == 'exp' or answer == 'explore':
            explore_loop()
            continue

        # go into generation mode
        if answer == 'g' or answer == 'gen' or answer == 'generate':
            generate_loop()
            continue

        # skip this problem
        if answer == 's' or answer == 'skip':
            prob = get_problem()
            continue

        # check the answer
        passed, msg = prob.check_answer(answer)
        # print repr(safe_eval(answer))
        print msg
        if passed:
            ncorrect += 1
            print 'Got %d correct' % ncorrect
            prob = get_problem()

    print
    print 'Goodbye! :)'
    print 'You answered %d problems' % ncorrect


if __name__ == '__main__':
    # initialize readline to get nice editing, tab-completion and
    # history
    readline.read_init_file('.readline')
    try:
        readline.read_history_file('.history')
    except IOError:
        pass

    # load corpora
    corpus = load_corpus('corpus')

    # main loop
    main_loop()

    # save history
    readline.write_history_file('.history')
