import smbus # Import module to use Voltage Sensor
import time #import time module to use delay

address = 0x48 #// ADC Converter's 0x40~0x48
A0 = 0x40 #// Set the address of the A0 pin as input
A1 = 0x41 #// Set the address of the A1 pin as input
A2 = 0x42 #// Set address of A2 pin as input
A3 = 0x43 #// Set the address of the A3 pin as input

bus = smbus.SMBus(1)

while True:
     bus.read_byte_data(address, A0) #// measure the signal of pin A1
     value = bus.read_byte_data(address,A0) #// Measured signal
     v = abs(int((value-255)))*1.5
 #// Save in value variable
     print(v) 
 #// Convert the measured value to an accurate value through an equation
     time.sleep(0.2) #// Output measured value every 0.2 seconds
