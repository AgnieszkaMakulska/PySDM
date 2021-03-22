"""
Created at 01.06.2020

@author: Grzegorz Łazarski
"""

from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute


class MoleAmountImpl(ExtensiveAttribute):
    def __init__(self, particles_builder, *, name):
        super().__init__(particles_builder, name=name)


def MoleAmount(compound):
    def _constructor(pb):
        return MoleAmountImpl(pb, name='moles_'+compound)
    return _constructor
