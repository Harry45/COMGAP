from src.sampling import MCMC
import setting as st

sampling = MCMC(emulator=True, method='second', variance=False)
sampling.generate_samples(10000, 12, 'emulator_second_' + str(st.NLHS) + '_1')
sampling.generate_samples(10000, 12, 'emulator_second_' + str(st.NLHS) + '_2')
