from enum import unique,Enum
from typing import List



@unique
class OperationType(Enum):
    Null    =   'Null'
    Mult    =   'Mult'
    Add     =   'Add'
    Both    =   'MultAdd'

Null = OperationType.Null
Mult = OperationType.Mult
Add  = OperationType.Add
Both = OperationType.Both


@unique
class DirectionType(Enum):
    Left    = 'left'
    Right   = 'right'
    Top     = 'top'
    Down    = 'down'


    _reverse_directions = {
        Left:  Right,
        Right: Left,
        Top:   Down,
        Down:  Top
    }

    @property
    def reverse(self):
        if self == Left:    return Right
        elif self == Right: return Left
        elif self == Top:   return Down
        elif self == Down:  return Top
        

Left  = DirectionType.Left
Right = DirectionType.Right
Top   = DirectionType.Top
Down  = DirectionType.Down


@unique
class StationaryType(Enum):
    OutputStationary = 'Output Stationary'
    WeightStationary = 'Weight Stationary'

OutputStationary = StationaryType.OutputStationary
WeightStationary = StationaryType.WeightStationary





def size_format(b):
    b = b/8
    if b < 1024:
        return '%i' % b + 'B'
    elif 1024 <= b < 1024*1024:
        return '%.1f' % float(b/1024) + 'KB'
    elif 1024*1024 <= b < 1024*1024*1024:
        return '%.1f' % float(b/1024/1024) + 'MB'
    elif 1024*1024*1024 <= b < 1024*1024*1024*1024:
        return '%.1f' % float(b/1024/1024/1024) + 'GB'
    elif 1024*1024*1024*1024 <= b:
        return '%.1f' % float(b/1024/1024/1024/1024) + 'TB'
