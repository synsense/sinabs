#  Copyright (c) 2019-2019     aiCTX AG (Sadique Sheik, Qian Liu).
#
#  This file is part of sinabs
#
#  sinabs is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sinabs is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with sinabs.  If not, see <https://www.gnu.org/licenses/>.

def test_inputlayer():
    from tensorflow import keras

    inpLayer = keras.layers.InputLayer(input_shape=(5, 20, 20))
    kerasConf = inpLayer.get_config()

    from sinabs.layers import get_input_shape_from_keras_conf

    input_shape = get_input_shape_from_keras_conf(kerasConf)
    assert input_shape == (5, 20, 20)
