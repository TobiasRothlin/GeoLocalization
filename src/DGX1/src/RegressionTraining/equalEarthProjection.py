import torch

def equal_earth_projection(lat, lon):
        """
        Project the lat and lon to the equal earth projection
        :param lat: The latitude
        :param lon: The longitude
        :return: The projected x and y
        """
        A_1 = 1.340264
        A_2 = -0.081106
        A_3 = 0.000893
        A_4 = 0.003796

        omega = torch.arcsin((torch.sqrt(3)/2) * torch.cos(lat))

        x = (2*torch.sqrt(3)) * lon * torch.cos(omega)
        x /= 3*(9*A_4*omega**8 + 7*A_3*omega**6 + 3*A_2*omega**2 + A_1)

        y = A_4*omega**9 + A_3*omega**7 + A_2*omega**3 + A_1*omega

        return x, y
    
def inverse_equal_earth_projection(x, y):
    pass


if __name__ == "__main__":
    
    random_lat = torch.rand(1) * 180 - 90
    random_lon = torch.rand(1) * 360 - 180
    print(f"Random lat: {random_lat}, Random lon: {random_lon}")

    x, y = equal_earth_projection(random_lat, random_lon)

    print(f"x: {x}, y: {y}")

    lat, lon = inverse_equal_earth_projection(x, y)

    print(f"Lat: {lat}, Lon: {lon}")


