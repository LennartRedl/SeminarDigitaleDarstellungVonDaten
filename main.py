import numpy as np
import matplotlib.pyplot as plt

def create_signal(t, components):
    """Create a signal based on sine and cosine components."""
    signal = np.zeros_like(t)
    for amplitude, frequency, type in components:
        if type == 'sine':
            signal += amplitude * np.sin(2 * np.pi * frequency * t)
        elif type == 'cosine':
            signal += amplitude * np.cos(2 * np.pi * frequency * t)
    return signal

def to_binary(value, bits=8):
    """Convert a quantized value to a binary string."""
    max_val = 2**bits - 1
    int_value = int((value + 0.5) * max_val)  # Scale and convert to integer
    return format(int_value, f'0{bits}b')

def calculate_metrics(original, quantized):
    """Calculate SNR and MSE."""
    noise = original - quantized
    mse = np.mean(noise**2)
    signal_power = np.mean(original**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    return snr, mse

def main():
    # User input for creating the wave
    components = []
    while True:
        response = input("Do you want to add a sine or cosine component? (type 'done' to finish) ")
        if response.lower() == 'done':
            break
        type = response.lower()
        amplitude = float(input("Amplitude of the component: "))
        frequency = float(input("Frequency of the component (Hz): "))
        components.append((amplitude, frequency, type))
    
    if not components:
        print("No components entered. Exiting program.")
        return
    max_frequency = max(comp[1] for comp in components)
    
    sampling_rate = 2 * max_frequency + 1
    duration = 2  # seconds
    t_sampling = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    t_plot = np.linspace(0, duration, 1000, endpoint=False)

    original_signal = create_signal(t_plot, components)
    quantized_signal = np.round(create_signal(t_sampling, components) * (2**8)) / (2**8)
    quantized_signal_plot = np.interp(t_plot, t_sampling, quantized_signal)  # Interpolate for plotting

    # Calculate SNR and MSE
    snr, mse = calculate_metrics(original_signal, quantized_signal_plot)

    # Convert quantized values to binary and create a bitstream
    bitstream = ''.join(to_binary(val) for val in quantized_signal)
    print("Bitstream (first 100 bits):", bitstream[:100])  # Display the first 100 bits for brevity

    reconstructed_signal = np.zeros_like(t_plot)
    for i, sample in enumerate(quantized_signal):
        reconstructed_signal += sample * np.sinc(sampling_rate * (t_plot - t_sampling[i]))

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(7, 10))

    # Plot for original signal
    axs[0].plot(t_plot, original_signal, label='Original Signal')
    axs[0].set_title(f'Original Signal - SNR: {snr:.2f} dB, MSE: {mse:.5f}')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)
    axs[0].legend()

    # Plot for reconstructed signal
    axs[1].plot(t_plot, reconstructed_signal, 'g', label='Reconstructed Signal', linestyle='--')
    axs[1].set_title('Reconstructed Signal')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True)
    axs[1].legend()

    # Combined plot
    axs[2].plot(t_plot, original_signal, label='Original Signal')
    axs[2].stem(t_sampling, quantized_signal, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal')
    axs[2].plot(t_plot, reconstructed_signal, 'g', label='Reconstructed Signal', linestyle='--')
    axs[2].set_title('Signal Sampling and Reconstruction')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Amplitude')
    axs[2].grid(True)
    axs[2].legend()

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    # Show the plot
    plt.show()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
