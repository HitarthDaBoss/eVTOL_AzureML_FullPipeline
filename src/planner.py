import heapq
import numpy as np

class EnergyAStar:
    def __init__(self, grid_shape=(100,100), res=1.0, obstacle=None, heuristic_w=1.0, energy_w=1.0):
        self.grid_shape = grid_shape
        self.res = res
        self.obstacle = np.zeros(grid_shape, dtype=np.uint8) if obstacle is None else obstacle
        self.h_w = heuristic_w
        self.e_w = energy_w

    def world_to_grid(self, p):
        x = int(p[0] / self.res)
        y = int(p[1] / self.res)
        return np.clip(x, 0, self.grid_shape[0]-1), np.clip(y, 0, self.grid_shape[1]-1)

    def grid_to_world(self, c):
        return (c[0] + 0.5) * self.res, (c[1] + 0.5) * self.res

    def neighbors(self, c):
        x,y = c
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
            nx,ny = x+dx, y+dy
            if 0 <= nx < self.grid_shape[0] and 0 <= ny < self.grid_shape[1]:
                if self.obstacle[nx,ny] == 0:
                    yield (nx,ny)

    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def energy_cost(self, a, b, wind=None):
        d = self.heuristic(a,b)
        wind_factor = (1.0 + 0.05 * (np.linalg.norm(wind) if wind is not None else 0.0))
        return d * wind_factor

    def plan(self, start, goal, z=15.0, wind=None):
        s = self.world_to_grid(start)
        g = self.world_to_grid(goal)
        openq = []
        heapq.heappush(openq, (0.0 + self.h_w * self.heuristic(s,g), 0.0, s, None))
        came_from = {}
        cost_so_far = {s:0.0}
        while openq:
            _, cost, current, _ = heapq.heappop(openq)
            if current == g:
                break
            for n in self.neighbors(current):
                move_cost = self.energy_cost(current,n,wind)
                new_cost = cost_so_far[current] + move_cost + self.e_w * move_cost
                if n not in cost_so_far or new_cost < cost_so_far[n]:
                    cost_so_far[n] = new_cost
                    pri = new_cost + self.h_w * self.heuristic(n,g)
                    heapq.heappush(openq, (pri, new_cost, n, current))
                    came_from[n] = current
        if g not in came_from:
            return np.zeros((0,3))
        path = []
        cur = g
        while cur != s:
            xw,yw = self.grid_to_world(cur)
            path.append((xw,yw,z))
            cur = came_from[cur]
        path.append(( (s[0]+0.5)*self.res, (s[1]+0.5)*self.res, z ))
        path.reverse()
        return np.array(path, dtype=np.float32)
