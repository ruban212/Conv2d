{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "daaf92d2-1bab-4be3-b8f2-cf6a8477ddb3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5101a41f-9cda-45b4-8e7d-c567769bb1db",
   "metadata": {},
   "outputs": [],
   "source": [
    "class RubanConv2D(nn.Module):\n",
    "    def __init__(self, kernel, stride=1, padding=0):\n",
    "        super(RubanConv2D, self).__init__()\n",
    "        self.kernel = kernel\n",
    "        self.stride = stride\n",
    "        self.padding = padding\n",
    "    \n",
    "    def forward(self, input_matrix):\n",
    "        if self.padding > 0:\n",
    "            input_pad = F.pad(input_matrix, (self.padding, self.padding, self.padding, self.padding), mode='constant')\n",
    "        else:\n",
    "            input_pad = input_matrix\n",
    "        \n",
    "        input_height, input_width = input_pad.shape\n",
    "        kernel_height, kernel_width = self.kernel.shape\n",
    "        \n",
    "        output_height = (input_height - kernel_height) // self.stride + 1\n",
    "        output_width = (input_width - kernel_width) // self.stride + 1\n",
    "        \n",
    "        output_matrix = torch.zeros((output_height, output_width), dtype=input_matrix.dtype)\n",
    "        \n",
    "        for i in range(0, output_height):\n",
    "            for j in range(0, output_width):\n",
    "                region = input_pad[..., i*self.stride:i*self.stride+kernel_height, j*self.stride:j*self.stride+kernel_width]\n",
    "                output_matrix[i, j] = torch.sum(region * self.kernel)\n",
    "        \n",
    "        return output_matrix"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
