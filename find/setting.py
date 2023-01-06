import copy

class Setting:
    alph = "ABCDEFGHIJK"
    def __init__(self, party, label, outcomes):
        self.party = party
        self.label = label
        self.outcomes = outcomes
        """ in case an independent outcome can be assigned """
        self.value = outcomes[0]
        self.iter = iter(self.outcomes)
        self.check()

    def check(self):
        if self.label==0 and len(self.outcomes)>1:
            raise Exception("""Invalid setting: Too many outcomes for setting with
            label 0. Label 0 is reserved for trivial
            measurements with exactly one outcome""")
        return 0

    def __iter__(self):
        return Setting_iterator(self)

    @property
    def party_symbol(self) -> str:
        return self.alph[self.party]

    @property
    def symbol(self):
        return self.party_symbol + "{}".format(self.label) 

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return other * self.value
        else:
            return self.value * other.value

    def __rmul__(self, other):
        return self.__mul__(other)

    def prod(*settings):
        t = 1
        for s in settings:
            t = t * s
        return t

    def __eq__(self, other):
        #careful, value is not compared, on purpose!
        assert isinstance(other, Setting)
        return self.party==other.party and self.label==other.label

    def __hash__(self):
        return hash((self.party, self.label))

    def __neq__(self, other):
        return not(self == other)

    def __str__(self):
        return self.symbol + "={}".format(self.value)

class Setting_iterator:
    def __init__(self, setting):
        self.setting = setting
        self.iter = iter(setting.outcomes)
    def __next__(self):
        newval = next(self.iter)
        newsett = copy.copy(self.setting)
        newsett.value = newval
        return newsett


