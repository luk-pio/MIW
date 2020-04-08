import random


class state:
    def __init__(self, message, t_states=None, t_probs=None):
        if t_states is None:
            t_states = []
        if t_probs is None:
            t_probs = []

        self.prob_sum = sum(t_probs)
        assert self.prob_sum <= 1, f'Tried to initialize state {message}, summed probabilities > 1!'

        self.message = message
        self.t_states = t_states
        self.t_probs = t_probs

    def add_transition(self, state, prob):
        new_prob_sum = self.prob_sum + prob
        assert new_prob_sum <= 1, f'The probabilities in state {self.message}, are greater than 1!'

        self.t_states.append(state)
        self.t_probs.append(prob)
        self.prob_sum = new_prob_sum

    def transition(self):
        states_with_self = self.t_states + [ self ]
        transitions_with_self = self.t_probs + [ 1 - self.prob_sum ]
        return random.choices(states_with_self, transitions_with_self)[0]

    def __repr__(self):
        return f'state({self.message})'


initial = state("Initial")
end = state("End")

win = state("Win", [end], [1])
lose = state("Lose", [end], [1])
draw = state("Draw", [initial], [1])
run = state("Run", [initial, lose], [0.7, 0.3])

fight_alice = state("Fight", [win, lose], [0.9, 0.1])
alice = state("Alice", [fight_alice, run], [0.7, 0.3])

fight_john = state("Fight", [win, lose], [0.25, 0.25])
john = state("John", [fight_john], [1])

fight_florence = state("Fight", [win, lose], [0.005, 0.98])
florence = state("Florence", [fight_florence, run], [0.2, 0.8])

initial.add_transition(alice, 0.2)
initial.add_transition(john, 0.5)
initial.add_transition(florence, 0.1)


def play(state):
    while state.message != "End":
        print(state.message)
        state = state.transition()


play(initial)
