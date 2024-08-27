{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "72pPbsF2RoGa"
      },
      "outputs": [],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import numpy as np"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "class RubanConv2D(nn.Module):\n",
        "    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):\n",
        "        super(RubanConv2D, self).__init__()\n",
        "\n",
        "        self.in_channels = in_channels\n",
        "        self.out_channels = out_channels\n",
        "\n",
        "        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)\n",
        "        self.stride = stride if isinstance(stride, tuple) else (stride, stride)\n",
        "        self.padding = padding if isinstance(padding, tuple) else (padding, padding)\n",
        "\n",
        "        self.bias = bias\n",
        "        self.weight = nn.Parameter(\n",
        "            torch.randn(out_channels, in_channels, *self.kernel_size) * np.sqrt(2.0 / (in_channels * np.prod(self.kernel_size)))\n",
        "        )\n",
        "        if bias:\n",
        "            self.bias = nn.Parameter(torch.zeros(out_channels))\n",
        "        else:\n",
        "            self.register_parameter('bias', None)\n",
        "\n",
        "    def forward(self, x):\n",
        "        if any(self.padding):\n",
        "            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]), mode='constant', value=0)\n",
        "\n",
        "        batch_size, _, input_height, input_width = x.shape\n",
        "        output_height = (input_height - self.kernel_size[0]) // self.stride[0] + 1\n",
        "        output_width = (input_width - self.kernel_size[1]) // self.stride[1] + 1\n",
        "\n",
        "        output = torch.zeros(batch_size, self.out_channels, output_height, output_width, device=x.device)\n",
        "\n",
        "        for i in range(0, output_height):\n",
        "            for j in range(0, output_width):\n",
        "                region = x[:, :, i * self.stride[0]:i * self.stride[0] + self.kernel_size[0], j * self.stride[1]:j * self.stride[1] + self.kernel_size[1]]\n",
        "                output[:, :, i, j] = torch.sum(region.unsqueeze(1) * self.weight, dim=(2, 3, 4))\n",
        "\n",
        "        if self.bias is not None:\n",
        "            output += self.bias.view(1, -1, 1, 1)\n",
        "\n",
        "        return output"
      ],
      "metadata": {
        "id": "Fg-ZWwDERrHP"
      },
      "execution_count": 2,
      "outputs": []
    }
  ]
}