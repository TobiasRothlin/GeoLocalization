def create_indipendent_batches(full_list, number_of_batches):
    batch_size = len(full_list) // number_of_batches
    remainder = len(full_list) % number_of_batches

    independent_batches = []
    start = 0

    for i in range(number_of_batches):
        end = start + batch_size + (1 if i < remainder else 0)
        independent_batches.append(full_list[start:end])
        start = end

    return independent_batches


def create_batches(full_list,batch_size):
    independent_batches = []
    start = 0

    for i in range(len(full_list)//batch_size):
        end = start + batch_size
        independent_batches.append(full_list[start:end])
        start = end

    if len(full_list) % batch_size != 0:
        independent_batches.append(full_list[start:])

    return independent_batches