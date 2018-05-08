class ExchangeResult(object):
    pass


class Exchange(object):
    def __init__(self, sender, receiver, descriptors):
        super(Exchange, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.descriptors = descriptors

    def single_exchange(self, image, message=None, state=None):
        sender_message, sender_message_dist = \
            self.sender(message, None, image, None)
        (stop_bit, stop_dist), (receiver_message, receiver_message_dist), y, new_state = \
            self.receiver(sender_message, state, None, self.descriptors)

        result = ExchangeResult()
        result.sender_message = sender_message
        result.sender_message_dist = sender_message_dist
        result.stop_bit = stop_bit
        result.stop_dist = stop_dist
        result.receiver_message = receiver_message
        result.receiver_message_dist = receiver_message_dist
        result.y = y
        result.new_state = new_state

        return result

    def exchange(self, image, length=10):
        results = []
        for step in range(length):
            if step == 0:
                result = self.single_exchange(image)
                results.append(result)
                continue
            result = self.single_exchange(image, results[-1].receiver_message, results[-1].new_state)
            results.append(result)
        return results
        