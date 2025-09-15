#!/usr/bin/env python

import random

from brisket import hot


def generate_random_dna(length):
    return ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length))


dna = generate_random_dna(512 * 1024)
print(hot(dna))