{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "fb53eedc-ab36-46ca-a2bd-d0319cd7bc3a",
   "metadata": {},
   "outputs": [
    {
     "ename": "ImportError",
     "evalue": "cannot import name 'RubanConv2D' from 'Conv2d_layers' (C:\\Users\\mruba\\Inbuilt Func\\Conv2d_layers.py)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[16], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mConv2d_layers\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m RubanConv2D\n",
      "\u001b[1;31mImportError\u001b[0m: cannot import name 'RubanConv2D' from 'Conv2d_layers' (C:\\Users\\mruba\\Inbuilt Func\\Conv2d_layers.py)"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "from Conv2d_layers import RubanConv2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "620d8078-e95a-4fc4-a362-d8a28b7f4d29",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'RubanConv2D' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[6], line 4\u001b[0m\n\u001b[0;32m      1\u001b[0m input_matrix \u001b[38;5;241m=\u001b[39m torch\u001b[38;5;241m.\u001b[39mtensor([[\u001b[38;5;241m1\u001b[39m, \u001b[38;5;241m2\u001b[39m, \u001b[38;5;241m3\u001b[39m], [\u001b[38;5;241m4\u001b[39m, \u001b[38;5;241m5\u001b[39m, \u001b[38;5;241m6\u001b[39m], [\u001b[38;5;241m7\u001b[39m, \u001b[38;5;241m8\u001b[39m, \u001b[38;5;241m9\u001b[39m]], dtype\u001b[38;5;241m=\u001b[39mtorch\u001b[38;5;241m.\u001b[39mfloat32)\n\u001b[0;32m      2\u001b[0m kernel \u001b[38;5;241m=\u001b[39m torch\u001b[38;5;241m.\u001b[39mtensor([[\u001b[38;5;241m1\u001b[39m, \u001b[38;5;241m0\u001b[39m], [\u001b[38;5;241m0\u001b[39m, \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m]], dtype\u001b[38;5;241m=\u001b[39mtorch\u001b[38;5;241m.\u001b[39mfloat32)\n\u001b[1;32m----> 4\u001b[0m ruban_conv2d \u001b[38;5;241m=\u001b[39m RubanConv2D(kernel, stride\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m, padding\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0\u001b[39m)\n\u001b[0;32m      5\u001b[0m output \u001b[38;5;241m=\u001b[39m ruban_conv2d(input_matrix)\n\u001b[0;32m      6\u001b[0m \u001b[38;5;28mprint\u001b[39m(output)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'RubanConv2D' is not defined"
     ]
    }
   ],
   "source": [
    "input_matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], dtype=torch.float32)\n",
    "kernel = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)\n",
    "\n",
    "ruban_conv2d = RubanConv2D(kernel, stride=1, padding=1)\n",
    "output = ruban_conv2d(input_matrix)\n",
    "print(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11dac145-8477-4af9-ad0a-b8375133c3c3",
   "metadata": {},
   "outputs": [],
   "source": []
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
